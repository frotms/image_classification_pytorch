# coding=utf-8

import os
# import tensorflow as tf
import torch
from tensorboardX import SummaryWriter


class Logger:
    """
    tensorboardX summary for pytorch
    """
    def __init__(self, config):
        self.config = config
        self.train_summary_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "train")
        self.validate_summary_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "val")
        if not os.path.exists(self.train_summary_dir):
            os.makedirs(self.train_summary_dir)
        if not os.path.exists(self.validate_summary_dir):
            os.makedirs(self.validate_summary_dir)
        self.train_summary_writer = SummaryWriter(self.train_summary_dir)
        self.validate_summary_writer = SummaryWriter(self.validate_summary_dir)


    # it can summarize scalars and images.
    def data_summarize(self, step, summarizer="train", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the validate one
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.validate_summary_writer
        if summaries_dict is not None:
            summary_writer.add_scalars('./', summaries_dict, step)
            # summary = tf.Summary()
            # for tag, value in summaries_dict.items():
            #     summary.value.add(tag=tag, simple_value=value)
            # summary_writer.add_summary(summary, step)
            # summary_writer.flush()


    def graph_summary(self, net, summarizer="train"):
        summary_writer = self.train_summary_writer if summarizer == "train" else self.validate_summary_writer
        input_to_model = torch.rand(1, self.config['img_height'], self.config['img_width'], self.config['num_channels'])
        summary_writer.add_graph(net, (input_to_model,))


    def close(self):
        self.train_summary_writer.close()
        self.validate_summary_writer.close()
