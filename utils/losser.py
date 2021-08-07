# coding=utf-8
# https://pytorch.org/docs/master/nn.html#loss-functions
import sys
import numpy as np
import torch
import torch.nn.functional as F
this_module = sys.modules[__name__]


def set_losser(losser_cfg, logits, onehot_labels, labels, model, **kwargs):
    lossers = []
    for every in losser_cfg["fn"]:
        every_loss = every["weight"] * getattr(this_module, every["name"].lower() + "_loss")(logits, onehot_labels, model, **kwargs)
        lossers.append(every_loss)
    # insert sum of all lossers to the last position
    sum_losser = 0
    for _loss in lossers:
        sum_losser += _loss
    lossers.append(sum_losser)
    return lossers

def cross_entropy_loss(pred, label, model, **kwargs):
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion.cuda()
    labels = torch.argmax(label, -1) # onehot to label index
    return criterion(pred, labels)

def mse_loss(pred, label, model, **kwargs):
    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion.cuda()

    return criterion(pred, label)

def l1_loss(pred, label, model, **kwargs):
    criterion = torch.nn.L1Loss()
    if torch.cuda.is_available():
        criterion.cuda()
    return criterion(pred, label)

def smooth_l1_loss(pred, label, model, **kwargs):
    criterion = torch.nn.SmoothL1Loss()
    if torch.cuda.is_available():
        criterion.cuda()
    return criterion(pred, label)

def norm_l1_loss(pred, label, model, **kwargs):
    l1_reg = 0
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
    return l1_reg

def norm_l2_loss(pred, label, model, **kwargs):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.sum(torch.abs(param))
    return l2_reg

def ctc_loss(pred, label, model, **kwargs):
    # https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
    # loss = ctc_loss(input, target, input_lengths, target_lengths)
    #  (T, N, C)(T,N,C) , where T = \text{input length}T=input length ,
    #  N = \text{batch size}N=batch size ,
    #  and C = \text{number of classes (including blank)}C=number of classes (including blank)
    x = pred.log_softmax(2)
    x = x.permute(1, 0, 2)

    T = 10
    N = pred.data.cpu().numpy().shape[0]
    # torch.LongTensor([T] * N)
    # torch.full(size=(N,), fill_value=T, dtype=torch.long)
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    if torch.cuda.is_available():
        criterion.cuda()

    return criterion(x, label, input_lengths, target_lengths)

def sigmoid_mse_loss(pred, label, model, **kwargs):
    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion.cuda()

    return criterion(F.sigmoid(pred), label)