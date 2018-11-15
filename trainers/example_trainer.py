# coding=utf-8
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from trainers.base_trainer import BaseTrainer
from utils import utils

class ExampleTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, config, logger):
        super(ExampleTrainer, self).__init__(model, train_loader, val_loader, config, logger)
        self.create_optimization()


    def train_epoch(self):
        """
        training in a epoch
        :return: 
        """
        # Learning rate adjustment
        self.learning_rate = self.adjust_learning_rate(self.optimizer, self.cur_epoch)
        self.train_losses = utils.AverageMeter()
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        # Set the model to be in training mode (for dropout and batchnorm)
        self.model.net.train()
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.train_step(batch_x_var, batch_y_var)

            # printer
            self.logger.log_printer.iter_case_print(self.cur_epoch, self.eval_train, self.eval_validate,
                                                    len(self.train_loader), batch_idx+1, self.train_losses.avg, self.learning_rate)

            # tensorboard summary
            if self.config['is_tensorboard']:
                self.logger.summarizer.data_summarize(batch_idx, summarizer="train", summaries_dict={"lr":self.learning_rate, 'train_loss':self.train_losses.avg})

        time.sleep(1)


    def train_step(self, images, labels):
        """
        training in a step
        :param images: 
        :param labels: 
        :return: 
        """
        # Forward pass
        infer = self.model.net(images)

        # label to one_hot
        # ids = labels.long().view(-1,1)
        # print(ids)
        # # one_hot_labels = torch.zeros(32, 2).scatter_(dim=1, index=ids, value=1.)

        # Loss function
        losses = self.get_loss(infer,labels)

        loss = losses.item()#.data[0]
        # measure accuracy and record loss
        prec1, prec5 = self.compute_accuracy(infer.data, labels.data, topk=(1, 5))
        self.train_losses.update(loss, images.size(0))
        self.train_top1.update(prec1[0], images.size(0))
        self.train_top5.update(prec5[0], images.size(0))
        # Optimization step
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.optimizer.module.zero_grad()
        else:
            self.optimizer.zero_grad()
        losses.backward()
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.optimizer.module.step()
        else:
            self.optimizer.step()


    def get_loss(self, pred, label):
        """
        compute loss
        :param pred: 
        :param label: 
        :return: 
        """
        criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
        if torch.cuda.is_available():
            criterion.cuda()
        return criterion(pred, label)


    def create_optimization(self):
        """
        optimizer
        :return: 
        """
        self.optimizer = torch.optim.Adam(self.model.net.parameters(),
                                          lr=self.config['learning_rate'], weight_decay=0) #lr:1e-4
        if torch.cuda.device_count() > 1:
            print('optimizer device_count: ',torch.cuda.device_count())
            self.optimizer = nn.DataParallel(self.optimizer,device_ids=range(torch.cuda.device_count()))
        """
        # optimizing parameters seperately
        ignored_params = list(map(id, self.model.net.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                            self.model.net.parameters())
        self.optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': self.model.net.fc.parameters(), 'lr': 1e-3}
            ], lr=1e-2, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)"""


    def adjust_learning_rate(self, optimizer, epoch):
        """
        decay learning rate
        :param optimizer: 
        :param epoch: the first epoch is 1
        :return: 
        """
        # """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
        # lr = lr_init
        # if epoch >= num_epochs * 0.75:
        #     lr *= decay_rate ** 2
        # elif epoch >= num_epochs * 0.5:
        #     lr *= decay_rate
        learning_rate = self.config['learning_rate'] * (self.config['learning_rate_decay'] ** ((epoch - 1) // self.config['learning_rate_decay_epoch']))
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = learning_rate
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        return learning_rate


    def compute_accuracy(self, output, target, topk=(1,)):
        """
        compute top-n accuracy
        :param output: 
        :param target: 
        :param topk: 
        :return: 
        """
        maxk = max(topk)
        batch_size = target.size(0)
        _, idx = output.topk(maxk, 1, True, True)
        idx = idx.t()
        correct = idx.eq(target.view(1, -1).expand_as(idx))
        acc_arr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_arr.append(correct_k.mul_(1.0 / batch_size))
        return acc_arr


    def evaluate_epoch(self):
        """
        evaluating in a epoch
        :return: 
        """
        self.eval_losses = utils.AverageMeter()
        self.eval_top1 = utils.AverageMeter()
        self.eval_top5 = utils.AverageMeter()
        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.net.eval()
        for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.evaluate_step(batch_x_var, batch_y_var)
            utils.view_bar(batch_idx+1, len(self.val_loader))


    def evaluate_step(self, images, labels):
        """
        evaluating in a step
        :param images: 
        :param labels: 
        :return: 
        """
        with torch.no_grad():
            infer = self.model.net(images)
            # label to one_hot
            # ids = labels.long().view(-1, 1)
            # one_hot_labels = torch.zeros(32, 2).scatter_(dim=1, index=ids, value=1.)

            # Loss function
            losses = self.get_loss(infer, labels)
            loss = losses.item()#losses.data[0]

        # measure accuracy and record loss
        prec1, prec5 = self.compute_accuracy(infer.data, labels.data, topk=(1, 5))

        self.eval_losses.update(loss, images.size(0)) # loss.data[0]
        self.eval_top1.update(prec1[0], images.size(0))
        self.eval_top5.update(prec5[0], images.size(0))
