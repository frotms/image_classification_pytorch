# coding=utf-8
# https://pytorch.org/docs/master/optim.html
# https://zhuanlan.zhihu.com/p/110018001
import torch
from torch.optim import lr_scheduler

def set_lr_scheduler(optimizer, cfg, niter):
    """Return a learning rate scheduler
        Parameters:
        optimizer -- 网络优化器
        lr_policy -- 学习率scheduler的名称: linear | step | plateau | cosine
    """
    lr_decay_type = cfg['lr_scheduler']['lr_decay_type'].lower()
    lr_init = cfg['lr_scheduler']['lr_init']
    lr_decay_rate = cfg['lr_scheduler']['lr_decay_rate']
    lr_decay_epochs = cfg['lr_scheduler']['lr_decay_epochs']

    if lr_decay_type == 'linear':
        raise NotImplementedError
        # def lambda_rule(epoch):
        #     lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        #     return lr_l
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_decay_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=niter, gamma=0.1)
    elif lr_decay_type == 'reduceplateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_decay_type == 'cosineannealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=0)
    elif lr_decay_type == 'multiplicative':
        raise NotImplementedError
    elif lr_decay_type == 'multistep':
        raise NotImplementedError
    elif lr_decay_type == 'exponential':
        raise NotImplementedError
    elif lr_decay_type == 'cyclic':
        raise NotImplementedError
    elif lr_decay_type == 'onecycle':
        raise NotImplementedError
    elif lr_decay_type == 'cosineannealingwarmrestarts':
        raise NotImplementedError
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_decay_type)
    return scheduler