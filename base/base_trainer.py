# coding=utf-8

class BaseTrainer:
    def __init__(self, logger, cfg, dataloader, modeler, sysDB):
        self.logger = logger
        self.cfg = cfg
        self.dataloader = dataloader
        self.modeler = modeler
        self.sysDB = sysDB

    def run(self):
        """
        Training logic for an epoch
        :return:
        """
        raise NotImplementedError

    def train_epoch(self):
        """
        implement the logic of train epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def evaluate_epoch(self):
        """
        implement the logic of the evaluate
         -loop ever the number of iteration in the config
        -add any summaries you want using the summary
        :return:
        """
        raise NotImplementedError

