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

import torch

class GeneralTrainer(BaseTrainer):
    def __init__(self, logger, cfg, dataloader, modeler, sysDB):
        super(GeneralTrainer, self).__init__(logger, cfg, dataloader, modeler, sysDB)
        self._init()


    def _init(self):
        self.cur_epoch = 0
        self.cur_batch = 0
        self.disable_printer = True if self.sysDB is not None else False
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

        niter = self.cfg['trainer']['num_epochs'] * len(self.dataloader.train_loader)

        self.optimizer = optimizer_module.set_optimizer(self.modeler.net, self.cfg)
        self.scheduler = lr_scheduler_module.set_lr_scheduler(self.optimizer, self.cfg, niter)

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
            # # save model
            # self.save()
            # evaluate
            self.evaluate_epoch()
            # # save best model after evaluation
            # self.save_best()

    def train_epoch(self):
        self.avgmeter_loss.reset()
        self.modeler.net.train()
        for batch_idx, item in enumerate(tqdm(self.dataloader.train_loader, disable=self.disable_printer)):
            try:
                batch_image, batch_label, batch_onehot_label = item
            except Exception as e:
                utils.printf(e, self.cfg['logger']['use_printf'])
                continue
            self.cur_batch = batch_idx + 1
            self.cur_lr = float(self.optimizer.param_groups[0]['lr'])
            self.train_step(batch_image, batch_onehot_label)

            # print
            if (self.cur_batch % self.cfg['logger']['log_step'] == 0 \
                    or self.cur_batch == len(self.dataloader.train_loader)):
                log_info = ' epoch: {}, lr: {:.7f}, average loss: {:.4f}'.format(self.cur_epoch,
                                                                                 self.cur_lr,
                                                                                 self.avgmeter_loss.avg)
                if self.cfg['logger']['use_printf']:
                    print(log_info)

    def train_step(self, batch_x, batch_y):
        if self.use_gpu:
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            # batch_input_var, batch_gt_var = Variable(batch_input), Variable(batch_gt)

        # forward pass
        infer = self.modeler.net(batch_x)

        # loss function
        loss_list = losser_module.set_losser(self.cfg['losser'], infer, batch_y, None, self.modeler.net)
        target_loss = loss_list[-1]  # the last one is total loss
        self.loss = target_loss # target_loss.item()  # .data[0]

        # save and print your loss info
        self.avgmeter_loss.update(self.loss, batch_x.size(0))

        self.optimizer.zero_grad()

        target_loss.backward()

        self.optimizer.step()

        self.scheduler.step()


    def evaluate_epoch(self):
        self.modeler.net.eval()
        if self.cur_epoch % self.cfg["evaluator"]["every_evaluate_epoch"] == 0 and self.cfg["evaluator"][
            "testset_evaluate"]:
            self._evaluate_epoch()


    def _evaluate_epoch(self):
        if self.cfg["evaluator"]["trainset_evaluate"]:
            self.train_metrics = self._evaluate(self.dataloader.evaluate_train_loader)
            print("epoch: {}, lr: {:.7f}, train_metrics: {}".format(self.cur_epoch, self.cur_lr,self.train_metrics))
        if self.cfg["evaluator"]["testset_evaluate"]:
            self.test_metrics = self._evaluate(self.dataloader.evaluate_test_loader)
            print("epoch: {}, lr: {:.7f}, test_metrics: {}".format(self.cur_epoch, self.cur_lr,self.test_metrics))

    def _evaluate(self, data_loader):
        preds, gts, labels = [], [], []
        for batch_idx, item in enumerate(tqdm(data_loader, disable=self.disable_printer)):
            try:
                batch_image, batch_label, batch_onehot_label = item
            except Exception as e:
                utils.printf(e, self.cfg['logger']['use_printf'])
                continue
            if self.use_gpu:
                batch_image = batch_image.cuda(non_blocking=True)

            with torch.no_grad():
                pred_e = self.modeler.net(batch_image)
            gt_e = batch_onehot_label
            preds += [pred_e.cpu().numpy()]
            gts += [gt_e.cpu().numpy()]

            # other info
            labels += batch_label.tolist()

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        return metrics_module.set_metrics(self.cfg["evaluator"]["metrics"], preds, gts)

    def save(self):
        save_path = self.cfg['saver']['save_dir']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, self.cfg['saver']['save_name'])
        torch.save(self.modeler.net.state_dict(), save_name)
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