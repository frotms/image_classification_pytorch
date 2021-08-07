# coding=utf-8


from __future__ import print_function
import numpy as np
import os, sys
import importlib
import cv2

np.seterr(divide='ignore',invalid='ignore')

def get_module(name, module):
    """Dynamically loading module"""
    return importlib.import_module(name + "." + module)

def get_instance(name, config, *args):
    """Dynamically loading instance"""
    return getattr(importlib.import_module(name + "." + config[name.split(".")[-1]]["module"]), config[name.split(".")[-1]]["class_name"])(*args)


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gpus_str_to_list(gpus_index_str):
    gpus_list = []
    _str = gpus_index_str.replace(' ','')
    _list = _str.split(',')
    for i, val in enumerate(_list):
        try:
            _val = int(val)
            gpus_list.append(_val)
        except:
            continue
    return gpus_list


def cv_imread(path):
    try:
        with open(path, "rb") as f:
            image = np.array(bytearray(f.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        image = None
    return image


def printf(msg, use_print):
    if use_print:
        print(msg)


def json_reader(path):
    import json
    cfg_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("{} is not existed.".format(cfg_path))

    with open(cfg_path, 'r', encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def yaml_reader(path):
    import yaml
    cfg_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("{} is not existed.".format(cfg_path))

    # with open(cfg_path, 'r', encoding="utf-8") as f:
    #     cfg = yaml.safe_load(f)
    cfg = yaml.load(open(cfg_path, 'rb'), Loader=yaml.Loader)
    return cfg

def cfg_reader(path):
    _, ext = os.path.splitext(path)
    assert ext.lower() in ['.json', '.yml', '.yaml']
    cfg = json_reader(path) if ext.lower() == ".json" else yaml_reader(path)
    return cfg