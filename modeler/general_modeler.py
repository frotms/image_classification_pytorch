# coding=utf-8

import os
import importlib
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
PROJ_NAME = os.path.basename(os.path.dirname(__dir__))

from ..base.base_modeler import BaseModeler
from ..utils import utils, pt_utils

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

class GeneralModeler(BaseModeler):
    def __init__(self, logger, cfg):
        super(GeneralModeler, self).__init__(logger, cfg)
        self.losses = []
        self.use_gpu = pt_utils.cuda_is_available()
        self.device_count = pt_utils.cuda_device_count()
        self.m = utils.get_module('{}.nets'.format(PROJ_NAME), self.cfg['modeler']['model']['module'])
        self.n = getattr(self.m, self.cfg['modeler']['model']['model_arch'])
        self.build_model()

        # load weights here

        # if self.use_gpu:
        #     self.net.cuda()

    def build_model(self):
        output_type = self.cfg['modeler']['model']['output_layer']
        num_classes = self.cfg['modeler']['model']['num_classes']
        params = {'num_classes': num_classes,
                  'pretrained': False,
                  }

        self.net = self.n(**params)

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError