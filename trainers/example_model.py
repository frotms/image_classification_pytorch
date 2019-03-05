# coding=utf-8
import os
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from trainers.base_model import BaseModel
from nets.net_interface import NetModule

class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.config = config
        self.interface = NetModule(self.config['model_module_name'], self.config['model_net_name'])
        self.create_model()


    def create_model(self):
        self.net = self.interface.create_model(num_classes=self.config['num_classes'])
        if torch.cuda.is_available():
            self.net.cuda()


    def load(self):
        # train_mode: 0:from scratch, 1:finetuning, 2:update
        # if not update all parameters:
        # for param in list(self.net.parameters())[:-1]:    # only update parameters of last layer
        #    param.requires_grad = False
        train_mode = self.config['train_mode']
        if train_mode == 'fromscratch':
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net)
            if torch.cuda.is_available():
                self.net.cuda()
            print('from scratch...')

        elif train_mode == 'finetune':
            self._load()
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net,device_ids=range(torch.cuda.device_count()))
            if torch.cuda.is_available():
                self.net.cuda()
            print('finetuning...')

        elif train_mode == 'update':
            self._load()
            print('updating...')

        else:
            ValueError('train_mode is error...')


    def _load(self):
        _state_dict = torch.load(os.path.join(self.config['pretrained_path'], self.config['pretrained_file']),
                                map_location=None)
        # for multi-gpus
        state_dict = OrderedDict()
        for item, value in _state_dict.items():
            if 'module' in item.split('.')[0]:
                name = '.'.join(item.split('.')[1:])
            else:
                name = item
            state_dict[name] = value
        # for handling in case of different models compared to the saved pretrain-weight
        model_dict = self.net.state_dict()
        same = {k: v for k, v in state_dict.items() if \
                (k in model_dict and model_dict[k].size() == v.size())}  # or (k not in state_dict)}
        diff = {k: v for k, v in state_dict.items() if \
                (k in model_dict and model_dict[k].size() != v.size()) or (k not in model_dict)}
        print('diff: ', [i for i, v in diff.items()])
        model_dict.update(same)
        self.net.load_state_dict(model_dict)
