# coding=utf-8

import copy
import os
import numpy as np
import cv2
import random
import time
import uuid
import torch
from PIL import Image
from collections import OrderedDict
from ..base.base_data_loader import BaseDataLoader
from .data_processor import DataProcessor
from ..utils import utils
from .samplers.distributed_sampler import InferenceSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler
from ..utils.dist import (
    get_local_rank,
    get_rank,
    get_local_size,
    get_world_size,
    synchronize,

)

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

class GeneralDataLoader(BaseDataLoader):
    def __init__(self, logger, cfg=None, is_distributed=False):
        super(GeneralDataLoader, self).__init__(logger, cfg, is_distributed)
        self.local_rank = get_local_rank()
        self.init()

    def init(self):
        # train_dataset
        train_gt_list = self._get_groundtruth_from_txt(self.cfg["data_loader"]["train_image_root_dir"],
                                                       self.cfg["data_loader"]["train_file"],
                                                       self.cfg["data_loader"]["file_label_separator"])
        self.train_dataset = DefineDataset(self.logger, self.cfg, train_gt_list, transform=None, is_data_aug=True)
        # use_balanced_sampler = False
        # batch_size = sum(self.cfg["trainer"]["batch_size"])
        # if use_balanced_sampler:
        #     raise NotImplementedError
        # else:
        #     num_samples = int(len(self.train_dataset) * self.cfg["data_loader"]["sample_rate"])
        #     if num_samples < batch_size:
        #         self.logger.error("sample_rate is too small to use a batch-processing")
        #         raise ValueError("sample_rate is too small to use a batch-processing")
        #     _sampler = RandomSampler(self.train_dataset, num_samples=num_samples, replacement=True)

        batch_size = sum(self.cfg["trainer"]["batch_size"])
        if self.is_distributed:
            batch_size = batch_size // get_world_size()

        # Partition dataset among workers using DistributedSampler
        _sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.train_dataset,
                                                                   num_replicas=get_world_size(),
                                                                   rank=get_local_rank(),
                                                                   shuffle=True,
                                                                   drop_last=True,
                                                                   )

        train_alignCollate = AlignCollate(cfg=self.cfg, is_training=True)

        dataloader_kwargs = {"batch_size": batch_size,
                             "num_workers": self.cfg['data_loader']['num_workers'],
                             "collate_fn": train_alignCollate,
                             "pin_memory": True,
                             }
        dataloader_kwargs["sampler"] = _sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       **dataloader_kwargs
                                       )


        # evaluate train dataset
        test_alignCollate = AlignCollate(cfg=self.cfg, is_training=False)
        if self.cfg["evaluator"]["trainset_evaluate"]:
            self.evaluate_train_dataset = DefineDataset(self.logger, self.cfg, train_gt_list, transform=None,
                                                        is_data_aug=False)

            if self.is_distributed:
                batch_size = sum(self.cfg["evaluator"]["batch_size"]) // get_world_size()
                sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset, shuffle=False, drop_last=False,
                )
            else:
                batch_size = sum(self.cfg["evaluator"]["batch_size"])
                sampler = torch.utils.data.SequentialSampler(self.train_dataset)

            dataloader_kwargs = {"batch_size": batch_size,
                                 "shuffle": False,
                                 "num_workers": self.cfg['evaluator']['num_workers'],
                                 "pin_memory": True,
                                 "sampler": sampler,
                                 "collate_fn": test_alignCollate,
                                 "drop_last": False,
                                 }
            self.evaluate_train_loader = DataLoader(dataset=self.train_dataset,
                                                    **dataloader_kwargs
                                                    )

        # evaluate test dataset
        if self.cfg["evaluator"]["testset_evaluate"]:
            test_gt_list = self._get_groundtruth_from_txt(self.cfg["data_loader"]["test_image_root_dir"],
                                                          self.cfg["data_loader"]["test_file"],
                                                          self.cfg["data_loader"]["file_label_separator"])

            self.evaluate_test_dataset = DefineDataset(self.logger, self.cfg, test_gt_list, transform=None,
                                                       is_data_aug=False)
            if self.is_distributed:
                batch_size = sum(self.cfg["evaluator"]["batch_size"]) // get_world_size()
                # torch.utils.data.distributed.DistributedSampler cannot work when len(dataset) is odd
                # here use InferenceSampler
                sampler = InferenceSampler(size=len(self.evaluate_test_dataset))
            else:
                batch_size = sum(self.cfg["evaluator"]["batch_size"])
                sampler = torch.utils.data.SequentialSampler(self.evaluate_test_dataset)
            print(batch_size, get_world_size())
            dataloader_kwargs = {"batch_size": batch_size,
                                 "shuffle": False,
                                 "num_workers": self.cfg['evaluator']['num_workers'],
                                 "pin_memory": True,
                                 "sampler": sampler,
                                 "collate_fn": test_alignCollate,
                                 "drop_last": False,
                                 }
            self.evaluate_test_loader = DataLoader(dataset=self.evaluate_test_dataset,
                                                   **dataloader_kwargs,
                                                   )


    def _get_groundtruth_from_txt(self, root_dir, txt, file_label_separator):
        """

        :param root_dir:
        :param txt:
        :param file_label_separator:
        :return:
        """
        result = []
        with open(txt, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip('\n\r').strip('\n').strip('\r')
                words = line.split(file_label_separator)
                if len(words) <= 1:
                    msg = "item {}: format error in txt file: {}".format(idx, txt)
                    utils.printf(msg, self.cfg['logger']['use_printf'])
                    continue
                full_file_path = os.path.join(root_dir, words[0])
                label_str = words[1].split(",")
                label = []
                image_id = words[2] if len(words) == 3 else idx

                for i, j in enumerate(label_str):
                    # 如果txt文件某个标签有问题，则continue该标签id
                    try:
                        label.append(int(j))
                    except Exception as e:
                        label = []
                        msg = "txt: line: {}, filename: {}, groundtruth: (label_id: {}, label_val: {}) error!".format(
                            image_id, words[0], i, j)
                        utils.printf(msg, self.cfg['logger']['use_printf'])
                        continue

                result.append({"id": image_id, "image_path": full_file_path, "label": label})
        return result


class DefineDataset(Dataset):
    def __init__(self, logger, cfg, ground_truth_list, transform=None, is_train_set=True, is_data_aug=True, sysDB=None):
        super(DefineDataset, self).__init__()
        self.logger = logger
        self.cfg = cfg
        self.sysDB = sysDB
        self.ground_truth_list = ground_truth_list
        self.DataProcessor = DataProcessor(self.cfg)
        self.transform = transform
        self.is_train_set = is_train_set
        self.is_data_aug = is_data_aug
        self.is_multilabel = False

    def __getitem__(self, index):
        cur_item_dict = self.ground_truth_list[index]
        path, label = cur_item_dict['image_path'], cur_item_dict['label']
        _root_dir = self.cfg['data_loader']['train_image_root_dir'] if self.is_train_set else self.cfg['data_loader']['test_image_root_dir']
        image = self.self_defined_loader(os.path.join(_root_dir, path))
        onehot_label = [0.] * self.cfg["modeler"]["model"]["num_classes"]
        for i in cur_item_dict["label"]:
            try:
                onehot_label[int(i)] = 1.
            except Exception as e:
                utils.printf(e, self.cfg['logger']['use_printf'])
                continue
        if self.transform is not None:
            image = self.transform(image)

        return image, label, onehot_label

    def __len__(self):
        return len(self.ground_truth_list)

    def self_defined_loader(self, filename):
        image = self.DataProcessor.image_loader(filename)
        image = self.DataProcessor.image_resize(image)
        if self.is_data_aug:
            image = self.DataProcessor.data_aug(image)
        image = self.DataProcessor.input_norm(image)
        return image


class AlignCollate:
    def __init__(self, cfg, is_training=False):
        self.cfg = cfg
        self.is_training = is_training

    def __call__(self, batch):
        image, label, onehot_label = zip(*batch)
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.int64)
        onehot_label = np.array(onehot_label, dtype=np.float32)
        image = np.transpose(image, (0,3,1,2))
        return torch.from_numpy(image), torch.from_numpy(label), torch.from_numpy(onehot_label)
        # return image, label, onehot_label

