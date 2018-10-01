# -*- coding: utf-8 -*-

import os
import cv2
import copy
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_loader.data_processor import DataProcessor


class PyTorchDataset(Dataset):
    def __init__(self, txt, config, transform=None, loader = None,
                 target_transform=None,  is_train_set=True):
        self.config = config
        imgs = []
        with open(txt,'r') as f:
            for line in f:
                line = line.strip('\n\r').strip('\n').strip('\r')
                words = line.split(self.config['file_label_separator'])
                # single label here so we use int(words[1])
                imgs.append((words[0], int(words[1])))

        self.DataProcessor = DataProcessor(self.config)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        _root_dir = self.config['train_data_root_dir'] if self.is_train_set else self.config['val_data_root_dir']
        image = self.self_defined_loader(os.path.join(_root_dir, fn))
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return len(self.imgs)


    def self_defined_loader(self, filename):
        image = self.DataProcessor.image_loader(filename)
        image = self.DataProcessor.image_resize(image)
        if self.is_train_set and self.config['data_aug']:
            image = self.DataProcessor.data_aug(image)
        image = self.DataProcessor.input_norm(image)
        return image


def get_data_loader(config):
    """
    
    :param config: 
    :return: 
    """
    train_data_file = config['train_data_file']
    test_data_file = config['val_data_file']
    batch_size = config['batch_size']
    num_workers =config['dataloader_workers']
    shuffle = config['shuffle']

    if not os.path.isfile(train_data_file):
        raise ValueError('train_data_file is not existed')
    if not os.path.isfile(test_data_file):
        raise ValueError('val_data_file is not existed')

    train_data = PyTorchDataset(txt=train_data_file,config=config,
                           transform=transforms.ToTensor(), is_train_set=True)
    test_data = PyTorchDataset(txt=test_data_file,config=config,
                                transform=transforms.ToTensor(), is_train_set=False)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader



