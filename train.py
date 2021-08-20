#coding:utf-8
import os
import sys
import time
import random
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.dirname(__dir__))
import importlib
PROJ_NAME = os.path.basename(__dir__)
utils = importlib.import_module('{}.utils.utils'.format(PROJ_NAME))
pt_utils = importlib.import_module('{}.utils.pt_utils'.format(PROJ_NAME))
comm = importlib.import_module('{}.utils.dist'.format(PROJ_NAME))
launch = importlib.import_module('{}.utils.launch'.format(PROJ_NAME))
setup_env = importlib.import_module('{}.utils.setup_env'.format(PROJ_NAME))


import torch
from datetime import timedelta

class AiImageCls:
    def __init__(self, cfg_path, args):
        self.proj_name = PROJ_NAME
        self.cfg = self.cfg_reader(cfg_path)
        self.args = args
        self.init()


    def init(self):
        # gpu_num = str(self.cfg["gpu_setter"]['gpu_id'])
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12345'
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        # # os.environ["NCCL_DEBUG"] = "INFO"
        # self.init_dist()

        self.trainer = utils.get_instance("{}.trainer".format(self.proj_name), self.cfg, self.cfg, self.args)

        # # create logger
        # # self.logger = utils.get_instance("{}.logger".format(self.proj_name), self.cfg, self.cfg, self.sysDB)
        # self.logger = None
        #
        # # create your data generator
        # self.dataloader = utils.get_instance("{}.data_loader".format(self.proj_name), self.cfg, self.logger, self.cfg, self.sysDB)
        # # convert decay_epoch to decat_step
        # self.cfg["lr_scheduler"]["lr_decay_epochs"] = len(self.dataloader.train_loader) * max(
        #     self.cfg["lr_scheduler"]["lr_decay_epochs"], 1)
        # # create instance of the model you want
        # self.modeler = utils.get_instance("{}.modeler".format(self.proj_name), self.cfg, self.logger, self.cfg, self.sysDB)
        #
        # # create trainer and path all previous components to it
        # self.trainer = utils.get_instance("{}.trainer".format(self.proj_name), self.cfg, self.logger, self.cfg,
        #                             self.dataloader, self.modeler, self.sysDB)



    def run(self):
        self.trainer.run()

    def close(self):
        raise NotImplementedError

    def cfg_reader(self, cfg_path):
        """
        .yml. .yaml or .json
        """
        cfg = utils.cfg_reader(cfg_path)
        return cfg

def parse_args():
    import argparse
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    DEFAULT_CONFIG_PATH = os.path.join(__dir__, 'configs/config.yml')
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    return parser.parse_args()


def main(args):
    seed = 666
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # set environment variables for distributed training
    setup_env.configure_nccl()
    setup_env.configure_omp()
    torch.backends.cudnn.benchmark = True

    executor = AiImageCls(cfg_path=args.config, args=args)
    executor.run()






# def train_run(rank, world_size, ):
#     config = os.path.join(__dir__, 'configs/config.yml')
#     cfg_path = os.path.abspath(os.path.expanduser(config))
#     if not os.path.exists(cfg_path):
#         raise FileNotFoundError('{} is not existed.'.format(cfg_path))
#
#     now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))
#
#     print('----------------------------------------------------------------------')
#     print('Time: ' + now)
#     print('----------------------------------------------------------------------')
#     print('                    Now start ...')
#     print('----------------------------------------------------------------------')
#
#     executor = AiImageCls(cfg_path=cfg_path, rank=rank, world_size=world_size)
#     executor.run()
#     # executor.close()
#
#     print('----------------------------------------------------------------------')
#     print('                      All Done!')
#     print('----------------------------------------------------------------------')
#     print('Start time: ' + now)
#     print('Now time: ' + time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time())))
#     print('----------------------------------------------------------------------')


# def main_(args):
#     gpu_num = "0,1"#str(self.cfg["gpu_setter"]['gpu_id'])
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12345'
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
#     # os.environ["NCCL_DEBUG"] = "INFO"
#
#     pt_utils.mp.spawn(train_run,
#              args=(args.world_size, ),
#              nprocs=args.world_size,
#              join=True)

if __name__ == '__main__':
    gpu_num = "0,1"#str(self.cfg["gpu_setter"]['gpu_id'])
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    # os.environ["NCCL_DEBUG"] = "INFO"
    args = parse_args()
    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()
    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch.launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(args,),
    )