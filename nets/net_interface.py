# coding=utf-8
import os
from importlib import import_module

class NetModule(object):
    def __init__(self, module_name, net_name, **kwargs):
        self.module_name = module_name
        self.net_name = net_name
        self.m = import_module('nets.' + self.module_name)

    def create_model(self, **kwargs):
        """
        when use a pretrained model of imagenet, pretrained_model_num_classes is 1000
        :param kwargs: 
        :return: 
        """
        _model = getattr(self.m, self.net_name)
        model = _model(**kwargs)
        return model