# coding=utf-8

from __future__ import print_function
import os, sys
import logging


class BaseLogger:
    """
    self.log_writer.info("log")
    self.log_writer.warning("warning")
    self.log_writer.error("error")

    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.flush()
    """

    def __init__(self, cfg, sysDB):
        self.cfg = cfg
        self.sysDB = sysDB
        self.training_log_writer = self.__init("training", "{}_train.log".format(cfg['info']['task']))
        self.evaluation_log_writer = self.__init("evaluation", "{}_eval.log".format(cfg['info']['task']))
        self.err_writer = self.__init("exceptions", "{}_err.log".format(cfg['info']['task']))

    def __init(self, name, log_name):
        log_writer = logging.getLogger(name)
        log_writer.setLevel(logging.INFO)
        logger_default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "logs")
        logger_set_path = os.path.realpath(self.cfg["logger"]["save_dir"])
        self.log_dir = logger_set_path if os.path.exists(logger_set_path) else logger_default_path
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        handler = logging.FileHandler(os.path.join(self.log_dir, log_name), encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        logging_format = logging.Formatter(
            "@%(asctime)s [%(filename)s -- %(funcName)s]%(lineno)s : %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %A %H:%M:%S')
        handler.setFormatter(logging_format)
        log_writer.addHandler(handler)
        log_writer.info("========================")
        log_writer.info(self.cfg)
        return log_writer

    def error(self, log_info):
        self.err_writer.error(log_info)
        sys.stdout.flush()
