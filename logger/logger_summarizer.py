# coding=utf-8

import os, sys


class LoggerSummarizer:
    """
    summary for tensorboard
    """
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.summary_placeholders = {}
        self.summary_ops = {}

        logger_default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "logs")
        logger_set_path = os.path.realpath(self.cfg["logger"]["save_dir"])
        self.log_dir = logger_set_path if os.path.exists(logger_set_path) else logger_default_path

        self.summary_dir = os.path.join(self.log_dir, "summary")

        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # tensorboard code here


    # it can summarize scalars and images.
    def summarize(self, step, summaries_dict=None):
        raise NotImplementedError
