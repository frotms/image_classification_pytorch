# coding=utf-8

from __future__ import print_function
import os, sys
import numpy as np
import logging
from ..utils import utils
from ..base.base_logger import BaseLogger
from .logger_printer import LoggerPrinter


class GeneralLogger(BaseLogger):
    """
    self.log_writer.info("这是日记")
    self.log_writer.warning("这是警告日记")
    self.log_writer.error("这是错误日记")

    为防止ctrl+c中断程序导致日志不输出，或者不缓存，每次直接记录log时后面直接flush
    self.log_writer.info("这是日记")
    sys.stdout.flush()
    """

    def __init__(self, cfg, sysDB):
        super(GeneralLogger, self).__init__(cfg, sysDB)
        self.summarizer = None
        self.log_printer = LoggerPrinter()
        self.log_info = {}
