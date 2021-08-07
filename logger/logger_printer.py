# coding=utf-8

import os
import time

def red_text(text, bright=True):
    return '\033[%d;31;40m%s\033[0m' % (bright, text)


def green_text(text, bright=True):
    return '\033[%d;32;40m%s\033[0m' % (bright, text)


def yellow_text(text, bright=True):
    return '\033[%d;33;40m%s\033[0m' % (bright, text)


def blue_text(text, bright=True):
    return '\033[%d;34;40m%s\033[0m' % (bright, text)


def magenta_text(text, bright=True):
    return '\033[%d;35;40m%s\033[0m' % (bright, text)


def cyan_text(text, bright=True):
    return '\033[%d;36;40m%s\033[0m' % (bright, text)

highlight_format = [red_text, green_text, yellow_text, blue_text, magenta_text, cyan_text]

def highlight_text(str_list):
    """
    按不同颜色highlight输出
    :param str_list:
    :return:
    """
    print_text = ""
    for i, content in enumerate(str_list):
        highlight_idx = i % len(highlight_format)
        print_text += highlight_format[highlight_idx](content)
    return print_text


class LoggerPrinter:
    """
    打印类
    """
    def print_every_step(self, trainer_instance):
        raise NotImplementedError

    def print_every_evaluation(self, trainer_instance):
        raise NotImplementedError
