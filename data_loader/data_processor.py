# coding=utf-8

import random
import numpy as np
import cv2
import scipy.misc as misc
from data_loader.data_augmentation import DataAugmenters


##########################################################
# name:     DataProcessor
# breif:
#
# usage:
##########################################################
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.DataAugmenters = DataAugmenters(self.config)

    def image_loader(self, filename, **kwargs):
        """
        load your image data
        :param filename: 
        :return: 
        """
        image = cv2.imread(filename)
        if image is None:
            raise ValueError('image data is none when cv2.imread!')
        return image


    def image_resize(self, image, **kwargs):
        """
        resize your image data
        :param image: 
        :param kwargs: 
        :return: 
        """
        _size = (self.config['img_width'], self.config['img_height'])
        _resize_image = cv2.resize(image, _size)
        return _resize_image[:,:,::-1]  # bgr2rgb

    def input_norm(self, image, **kwargs):
        """
        normalize your image data
        :param image: 
        :return: 
        """
        return ((image - 127) * 0.0078125).astype(np.float32) # 1/128


    def data_aug(self, image, **kwargs):
        """
        augment your image data with DataAugmenters
        :param image: 
        :return: 
        """
        return self.DataAugmenters.run(image, **kwargs)


if __name__ == '__main__':

    print('done!')
