#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from importlib import import_module

class TagPytorchInference(object):

    def __init__(self, **kwargs):
        _input_size = kwargs.get('input_size',299)
        self.input_size = (_input_size, _input_size)
        self.gpu_index = kwargs.get('gpu_index', '0')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_index
        self.net = self._create_model(**kwargs)
        self._load(**kwargs)
        self.net.eval()
        self.transforms = transforms.ToTensor()
        if torch.cuda.is_available():
            self.net.cuda()

    def close(self):
        torch.cuda.empty_cache()


    def _create_model(self, **kwargs):
        module_name = kwargs.get('module_name','vgg_module')
        net_name = kwargs.get('net_name', 'vgg16')
        m = import_module('nets.' + module_name)
        model = getattr(m, net_name)
        net = model(**kwargs)
        return net


    def _load(self, **kwargs):
        model_name = kwargs.get('model_name', 'model.pth')
        model_filename = model_name
        state_dict = torch.load(model_filename, map_location=None)
        self.net.load_state_dict(state_dict)


    def run(self, image_data, **kwargs):
        _image_data = self.image_preproces(image_data)
        input = self.transforms(_image_data)
        _size = input.size()
        input = input.resize_(1, _size[0], _size[1], _size[2])
        if torch.cuda.is_available():
            input = input.cuda()
        logit = self.net(Variable(input))
        # softmax
        infer = F.softmax(logit, 1)
        return infer.data.cpu().numpy().tolist()


    def image_preproces(self, image_data):
        _image = cv2.resize(image_data, self.input_size)
        _image = _image[:,:,::-1]   # bgr2rgb
        return _image.copy()

if __name__ == "__main__":
    # # python3 inference.py --image test.jpg --module inception_resnet_v2_module --net inception_resnet_v2 --model model.pth
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', "--image", type=str, help='Assign the image path.', default=None)
    parser.add_argument('-module', "--module", type=str, help='Assign the module name.', default=None)
    parser.add_argument('-net', "--net", type=str, help='Assign the net name.', default=None)
    parser.add_argument('-model', "--model", type=str, help='Assign the net name.', default=None)
    parser.add_argument('-cls', "--cls", type=int, help='Assign the classes number.', default=None)
    parser.add_argument('-size', "--size", type=int, help='Assign the input size.', default=None)
    args = parser.parse_args()
    if args.image is None or args.module is None or args.net is None or args.model is None\
            or args.size is None or args.cls is None:
        raise TypeError('input error')
    if not os.path.exists(args.model):
        raise TypeError('cannot find file of model')
    if not os.path.exists(args.image):
        raise TypeError('cannot find file of image')
    print('test:')
    filename = args.image
    module_name = args.module
    net_name = args.net
    model_name = args.model
    input_size = args.size
    num_classes = args.cls
    image = cv2.imread(filename)
    if image is None:
        raise TypeError('image data is none')
    tagInfer = TagPytorchInference(module_name=module_name,net_name=net_name,
                                   num_classes=num_classes, model_name=model_name,
                                   input_size=input_size)
    result = tagInfer.run(image)
    print(result)
    print('done!')