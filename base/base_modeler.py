# coding=utf-8
import os

class BaseModeler:
    def __init__(self, logger, cfg):
        self.logger = logger
        self.cfg = cfg

    def save(self, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError