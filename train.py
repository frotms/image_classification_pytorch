#coding:utf-8
import os
import sys
import time
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.dirname(__dir__))
import importlib
PROJ_NAME = os.path.basename(__dir__)
utils = importlib.import_module('{}.utils.utils'.format(PROJ_NAME))



class AiImageCls:
    def __init__(self, cfg_path, sysDB=None):
        self.proj_name = PROJ_NAME
        self.cfg = self.cfg_reader(cfg_path)
        self.sysDB = sysDB
        self.use_dp = False
        self.use_ddp = False
        self.init()

    def init(self):
        gpu_num = str(self.cfg["gpu_setter"]['gpu_id'])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

        if self.cfg['gpu_setter']['gpu_parallel'].lower() == 'ddp':
            pass

        elif self.cfg['gpu_setter']['gpu_parallel'].lower() == 'dp':
            if len(utils.gpus_str_to_list(gpu_num)) > 1:
                self.use_dp = True
                pass
        else:
            pass

        # create logger
        # self.logger = utils.get_instance("{}.logger".format(self.proj_name), self.cfg, self.cfg, self.sysDB)
        self.logger = None

        # create your data generator
        self.dataloader = utils.get_instance("{}.data_loader".format(self.proj_name), self.cfg, self.logger, self.cfg, self.sysDB)
        # convert decay_epoch to decat_step
        self.cfg["lr_scheduler"]["lr_decay_epochs"] = len(self.dataloader.train_loader) * max(
            self.cfg["lr_scheduler"]["lr_decay_epochs"], 1)
        # create instance of the model you want
        self.modeler = utils.get_instance("{}.modeler".format(self.proj_name), self.cfg, self.logger, self.cfg, self.sysDB)
        if self.use_dp:
            pass

        # create trainer and path all previous components to it
        self.trainer = utils.get_instance("{}.trainer".format(self.proj_name), self.cfg, self.logger, self.cfg,
                                    self.dataloader, self.modeler, self.sysDB)

        if self.use_ddp:
            pass

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


if __name__ == '__main__':
    import argparse

    DEFAULT_CONFIG_PATH = './configs/config.yml' # yml, yaml or json
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, help='Assign the config path.', default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    cfg_path = os.path.abspath(os.path.expanduser(args.config))
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('{} is not existed.'.format(cfg_path))

    now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))

    print('----------------------------------------------------------------------')
    print('Time: ' + now)
    print('----------------------------------------------------------------------')
    print('                    Now start ...')
    print('----------------------------------------------------------------------')

    executor = AiImageCls(cfg_path=cfg_path)
    executor.run()
    # executor.close()

    print('----------------------------------------------------------------------')
    print('                      All Done!')
    print('----------------------------------------------------------------------')
    print('Start time: ' + now)
    print('Now time: ' + time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time())))
    print('----------------------------------------------------------------------')