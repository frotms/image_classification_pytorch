# coding=utf-8

import math
import os
from collections import OrderedDict
import numpy as np
import torch

class BaseModel:
    def __init__(self,config):
        self.config = config

    # save function thet save the checkpoint in the path defined in configfile
    def save(self):
        """
        implement the logic of saving model
        """
        print("Saving model...")
        save_path = self.config['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path,self.config['save_name'])
        state_dict = OrderedDict()
        for item, value in self.net.state_dict().items():
            if 'module' in item.split('.')[0]:
                name = '.'.join(item.split('.')[1:])
            else:
                name = item
            state_dict[name] = value
        torch.save(state_dict, save_name)
        print("Model saved: ", save_name)

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self):
        """
        implement the logic of loading model
        """
        raise NotImplementedError


    def build_model(self):
        """
        implement the logic of model
        """
        raise NotImplementedError