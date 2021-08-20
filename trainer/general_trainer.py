# coding=utf-8

import time
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from ..base.base_trainer import BaseTrainer
from ..utils import utils, pt_utils
from ..utils import losser as losser_module
from ..utils import lr_scheduler as lr_scheduler_module
from ..utils import metrics as metrics_module
from ..utils import optimizer as optimizer_module
from ..utils import ema, allreduce_norm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from ..utils.dist import (
    get_local_rank,
    get_rank,
    get_local_size,
    get_world_size,
    synchronize,
    is_main_process,
    reduce_value,
    all_gather,
)


class GeneralTrainer(BaseTrainer):
    def __init__(self, cfg, args):
        super(GeneralTrainer, self).__init__(cfg, args)
        self._init_dist()
        self._init_logger()
        self._init_data_loader()
        self._init_modeler()
        self._init_train()

    def _init_dist(self):
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        use_fp16 = self.cfg['modeler']['model']['fp16']
        self.amp_training = use_fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
        self.data_type = torch.float16 if use_fp16 else torch.float32
        self.use_model_ema = self.cfg['modeler']['model']['use_model_ema']

    def _init_logger(self):
        # create logger
        # self.logger = utils.get_instance("{}.logger".format(self.proj_name), self.cfg, self.cfg, self.sysDB)
        self.logger = None

    def _init_data_loader(self):
        # create your data generator
        self.dataloader = utils.get_instance("{}.data_loader".format(self.proj_name), self.cfg, self.logger, self.cfg, self.is_distributed)
        # convert decay_epoch to decat_step
        self.cfg["lr_scheduler"]["lr_decay_epochs"] = len(self.dataloader.train_loader) * max(
            self.cfg["lr_scheduler"]["lr_decay_epochs"], 1)

        # # prefetcher
        # from ..data_loader.data_prefetcher import DataPrefetcher
        # self.prefetcher = DataPrefetcher(self.dataloader.train_loader)

        # max_iter means iters per epoch
        self.max_iter = len(self.dataloader.train_loader)


    def _init_modeler(self):
        # model related init
        torch.cuda.set_device(self.local_rank)
        # create instance of the model you want
        self.modeler = utils.get_instance("{}.modeler".format(self.proj_name), self.cfg, self.logger, self.cfg)

        # load model weight

        self.modeler.net.to(self.device)


    def _init_train(self):
        self.cur_epoch = 0
        self.cur_batch = 0
        self.disable_printer = False
        # best evaluation
        self.best_evaluation = -1.
        # init metrics
        self.train_metrics = self._init_metrics()
        self.test_metrics = self._init_metrics()
        # init losses
        self.all_losses = [-1.]
        # avg total loss
        self.avgmeter_loss = utils.AverageMeter()
        # report_dict
        self.report_dict = {}
        # start_epoch
        self.start_epoch = self.cfg["trainer"]["start_epoch"]
        self.best_model_path = None

        self.use_gpu = pt_utils.cuda_is_available()
        self.device_count = pt_utils.cuda_device_count()
        self.cur_lr = self.cfg['lr_scheduler']['lr_init']
        niter = self.cfg['trainer']['num_epochs'] * len(self.dataloader.train_loader)

        self.optimizer = optimizer_module.set_optimizer(self.modeler.net, self.cfg)
        self.scheduler = lr_scheduler_module.set_lr_scheduler(self.optimizer, self.cfg, niter)

        if self.args.occupy:  # occupy
            pt_utils.occupy_mem(self.local_rank)

        if self.is_distributed:
            self.modeler.net = DDP(self.modeler.net, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ema.ModelEMA(self.modeler.net, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

    def _init_metrics(self):
        metrics = {}
        for every in self.cfg["evaluator"]["metrics"]["fn"]:
            metrics[every] = -1.
        return metrics

    def run(self):
        for cur_epoch in range(self.start_epoch, self.cfg["trainer"]["num_epochs"] + self.start_epoch):
            self.cur_epoch = cur_epoch
            # train
            self.train_epoch()

            synchronize()

            # # save model
            # self.save()
            # evaluate
            self.evaluate_epoch()
            # # save best model after evaluation
            # self.save_best()

    def train_epoch(self):
        self.avgmeter_loss.reset()
        self.modeler.net.train()
        # self.max_iter
        tqdm_print = self.disable_printer and is_main_process()
        for batch_idx, item in enumerate(tqdm(self.dataloader.train_loader, disable=tqdm_print)):
            try:
                batch_image, batch_label, batch_onehot_label = item
            except Exception as e:
                utils.printf(e, self.cfg['logger']['use_printf'])
                continue
            self.cur_batch = batch_idx + 1
            self.cur_lr = float(self.optimizer.param_groups[0]['lr'])
            self.train_step(batch_image, batch_onehot_label)

            # print
            if is_main_process():
                if (self.cur_batch % self.cfg['logger']['log_step'] == 0 \
                        or self.cur_batch == len(self.dataloader.train_loader)):
                    log_info = ' epoch: {}, lr: {:.7f}, average loss: {:.4f}'.format(self.cur_epoch,
                                                                                     self.cur_lr,
                                                                                     self.avgmeter_loss.avg)
                    if self.cfg['logger']['use_printf']:
                        print(log_info)

    def train_step(self, batch_x, batch_y):
        batch_x, batch_y = batch_x.to(self.data_type), batch_y.to(self.data_type)
        if self.use_gpu:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

        # forward pass
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            infer = self.modeler.net(batch_x)

        # loss function
        loss_list = losser_module.set_losser(self.cfg['losser'], infer, batch_y, None, self.modeler.net)
        target_loss = loss_list[-1]  # the last one is total loss
        reduce_loss = reduce_value(target_loss)
        self.loss = reduce_loss # target_loss.item()  # .data[0]

        # save and print your loss info
        self.avgmeter_loss.update(self.loss, batch_x.size(0))

        self.optimizer.zero_grad()
        # # backward
        # target_loss.backward()
        self.scaler.scale(target_loss).backward()
        # # optimizer step
        # self.optimizer.step()
        self.scaler.step(self.optimizer)
        # # scheduler step
        # self.scheduler.step()
        self.scaler.step(self.scheduler)

        self.scaler.update()

        # Not necessary to use a dist.barrier() to guard the file deletion below
        # as the AllReduce ops in the backward pass of DDP already served as
        # a synchronization

        if self.use_model_ema:
            self.ema_model.update(self.modeler.net)


    def evaluate_epoch(self):
        self.modeler.net.eval()
        if self.cur_epoch % self.cfg["evaluator"]["every_evaluate_epoch"] == 0 and self.cfg["evaluator"][
            "testset_evaluate"]:
            allreduce_norm.all_reduce_norm(self.modeler.net)
            self._evaluate_epoch()


    def _evaluate_epoch(self):
        if self.cfg["evaluator"]["trainset_evaluate"]:
            self.train_metrics = self._evaluate(self.dataloader.evaluate_train_loader)
            if is_main_process():
                print("epoch: {}, lr: {:.7f}, train_metrics: {}".format(self.cur_epoch, self.cur_lr,self.train_metrics))
        if self.cfg["evaluator"]["testset_evaluate"]:
            self.test_metrics = self._evaluate(self.dataloader.evaluate_test_loader)
            if is_main_process():
                print("epoch: {}, lr: {:.7f}, test_metrics: {}".format(self.cur_epoch, self.cur_lr,self.test_metrics))

    def _evaluate(self, data_loader):
        preds, gts, labels = [], [], []
        tqdm_print = self.disable_printer and is_main_process()
        for batch_idx, item in enumerate(tqdm(data_loader, disable=tqdm_print)):
            try:
                batch_image, batch_label, batch_onehot_label = item
            except Exception as e:
                utils.printf(e, self.cfg['logger']['use_printf'])
                continue
            batch_image, batch_onehot_label = batch_image.to(self.data_type), batch_onehot_label.to(self.data_type)
            if self.use_gpu:
                batch_image = batch_image.to(self.device)

            with torch.no_grad():
                pred_e = self.modeler.net(batch_image)
            gt_e = batch_onehot_label
            preds += [pred_e.cpu().numpy()]
            gts += [gt_e.numpy()]
            # other info
            labels += batch_label.tolist()
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        # dist.all_gather -> [gpu0_all, gpu1_all], gpu(n)_all shape: (num_samples, num_classes)
        # all_gather -> [(520,102), (519,102)]
        preds = all_gather(preds)
        gts = all_gather(gts)
        # concate example: (520,102), (519,102) -> (1039,102)
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        return metrics_module.set_metrics(self.cfg["evaluator"]["metrics"], preds, gts)

    def save(self):
        if is_main_process():
            save_path = self.cfg['saver']['save_dir']
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = os.path.join(save_path, self.cfg['saver']['save_name'])
            # torch.save(self.modeler.net.state_dict(), save_name)
            torch.save(self.modeler.net.module.state_dict() \
                           if hasattr(self.modeler.net, 'module') else self.modeler.net.state_dict(), save_name)
            print("Model saved: ", save_name)

    # def save(self):
    #     """
    #     implement the logic of saving model
    #     """
    #     print("Saving model...")
    #     save_path = self.cfg['saver']['save_dir']
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     save_name = os.path.join(save_path, self.cfg['saver']['save_name'])
    #     state_dict = OrderedDict()
    #     for item, value in self.modeler.net.state_dict().items():
    #         if 'module' in item.split('.')[0]:
    #             name = '.'.join(item.split('.')[1:])
    #         else:
    #             name = item
    #         state_dict[name] = value
    #     torch.save(state_dict, save_name)
    #     print("Model saved: ", save_name)

    def save_best(self):
        raise NotImplementedError