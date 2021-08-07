# coding=utf-8
# https://pytorch.org/docs/master/optim.html
# https://blog.csdn.net/shanglianlm/article/details/85019633
import torch

def set_optimizer(model, cfg, **kwargs):
    optimizer_type = cfg['optimizer']["type"].lower()
    if optimizer_type == 'adadelta':
        raise NotImplementedError
    elif optimizer_type == 'adagrad':
        raise NotImplementedError
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg['lr_scheduler']['lr_init'], weight_decay=0)  # lr:1e-4
    elif optimizer_type == 'adamw':
        raise NotImplementedError
    elif optimizer_type == 'sparseadam':
        raise NotImplementedError
    elif optimizer_type == "adamax":
        raise NotImplementedError
    elif optimizer_type == "asgd":
        raise NotImplementedError
    elif optimizer_type == "lbfgs":
        raise NotImplementedError
    elif optimizer_type == "rmsprop":
        raise NotImplementedError
    elif optimizer_type == "rprop":
        raise NotImplementedError
    elif optimizer_type == "sgd":
        raise NotImplementedError
    else:
        raise ValueError("optimizer error")

    if torch.cuda.device_count() > 1:
        print('optimizer device_count: ', torch.cuda.device_count())
        _optimizer = torch.nn.DataParallel(optimizer, device_ids=range(torch.cuda.device_count()))
        optimizer = _optimizer.module

    return optimizer