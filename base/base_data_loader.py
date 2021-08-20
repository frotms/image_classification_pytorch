# coding=utf-8
import copy


class BaseDataLoader:
    def __init__(self, logger, cfg=None, is_distributed=False):
        self.__init(logger, cfg, is_distributed)
        self.train_dataset = None
        self.evaluate_train_dataset = None
        self.evaluate_test_dataset = None
        self.train_loader = None
        self.evaluate_train_loader = None
        self.evaluate_test_loader = None    

    def __init(self, logger, cfg, is_distributed):
        self.logger = logger
        self.cfg = cfg
        self.is_distributed = is_distributed
