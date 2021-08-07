# coding=utf-8
import sys
import numpy as np
this_module = sys.modules[__name__]

def set_metrics(evaluator_metrics_cfg, preds, gts):
    metrics = {}
    for every in evaluator_metrics_cfg["fn"]:
        every_lower = every.lower()
        metrics[every] = getattr(this_module, "metric_" + every_lower)(preds, gts)
    return metrics

def metric_accuracy(pred, label, **kwargs):
    topk = 1
    nrows, ncols = pred.shape
    sort_arr = np.argsort(-pred)
    target_idx = np.where(label == True)

    correct = 0
    for i in range(len(target_idx[0])):
        correct = correct + 1 if target_idx[1][i] in sort_arr[i][:topk] else correct
    _acc = 1.0 * correct / nrows

    return _acc