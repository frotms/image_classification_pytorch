# coding=utf-8

import torch

def cuda_is_available():
    return torch.cuda.is_available()

def cuda_device_count():
    return torch.cuda.device_count()