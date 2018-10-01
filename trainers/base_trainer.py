# coding=utf-8
from __future__ import print_function
import os
import time
import torch
from torch.autograd import Variable


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_train = 0.
        self.eval_validate = 0.
        self.optimizer = None
        self.loss = None


    def train(self):
        total_epoch_num = self.config['num_epochs']
        if self.config['evaluate_before_train']:
            self.evaluate_epoch()
            self.eval_validate = self.eval_top1.avg
            print('\n')
        for cur_epoch in range(1, total_epoch_num+1):
            epoch_start_time = time.time()
            self.cur_epoch = cur_epoch
            self.train_epoch()
            self.evaluate_epoch()
            self.eval_train, self.eval_validate = self.train_top1.avg, self.eval_top1.avg
            # printer
            self.logger.log_printer.epoch_case_print(self.cur_epoch,
                                                     self.train_top1.avg, self.eval_top1.avg,
                                                     self.train_losses.avg,  self.eval_losses.avg,
                                                     time.time()-epoch_start_time)
            # save model
            self.model.save()
            # logger
            self.logger.write_info_to_logger(variable_dict={'epoch':self.cur_epoch, 'lr':self.learning_rate,
                                                            'train_acc':self.eval_train,'validate_acc':self.eval_validate,
                                                            'train_avg_loss':self.train_losses.avg,'validate_avg_loss':self.eval_losses.avg,
                                                            'gpus_index': self.config['gpu_id'],
                                                            'save_name': os.path.join(self.config['save_path'],
                                                                                       self.config['save_name']),
                                                            'net_name': self.config['model_net_name']})
            self.logger.write()
            # tensorboard summary
            if self.config['is_tensorboard']:
                self.logger.summarizer.data_summarize(self.cur_epoch, summarizer='train',
                                                 summaries_dict={'train_acc': self.eval_train, 'train_avg_loss': self.train_losses.avg})
                self.logger.summarizer.data_summarize(self.cur_epoch, summarizer='validate',
                                                 summaries_dict={'validate_acc': self.eval_validate,'validate_avg_loss': self.eval_losses.avg})
                # if self.cur_epoch == total_epoch_num:
                #     self.logger.summarizer.graph_summary(self.model.net)


    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        """
        raise NotImplementedError


    def train_step(self):
        """
        implement the logic of the train step
        """
        raise NotImplementedError


    def evaluate_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        """
        raise NotImplementedError


    def evaluate_step(self):
        """
        implement the logic of the train step
        """
        raise NotImplementedError


    def get_loss(self):
        """
        implement the logic of model loss
        """
        raise NotImplementedError


    def create_optimization(self):
        """
        implement the logic of the optimization
        """
        raise NotImplementedError