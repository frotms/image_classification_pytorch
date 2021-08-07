# coding=utf-8
import copy


class BaseDataLoader:
    def __init__(self, logger, cfg=None, sysDB=None):
        self.__init(logger, cfg, sysDB)
        self.train_dataset = None
        self.evaluate_train_dataset = None
        self.evaluate_test_dataset = None
        self.train_loader = None
        self.evaluate_train_loader = None
        self.evaluate_test_loader = None    

    def __init(self, logger, cfg, sysDB):
        self.logger = logger
        self.cfg = cfg
        self.sysDB = sysDB
