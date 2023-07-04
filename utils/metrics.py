from time import time
import torch
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
from neuralop.training.losses import LpLoss, H1Loss


METRIC_DICT = {
    'mae': mean_absolute_error,
    'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    'r2': r2_score,
    'mape': mean_absolute_percentage_error,
    'mse': mean_squared_error,
    # 'l2': lambda y_true, y_pred: torch.mean(torch.pow(y_true - y_pred, 2)),
    'l2': LpLoss(d=2, p=2),
    'h1': H1Loss(d=2),
    'MAE': mean_absolute_error,
    'RMSE': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    'R2': r2_score,
    'MAPE': mean_absolute_percentage_error,
    'MSE': mean_squared_error,
    'L2': LpLoss(d=2, p=2),
    'H1': H1Loss(d=2),
}
VALID_METRICS = list(METRIC_DICT.keys())


class AverageRecord(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    def __init__(self, metrics=['mae', 'r2'], split="valid"):
        self.metric_list = metrics
        self.start_time = time()
        self.split = split
        self.metrics = {metric: AverageRecord() for metric in self.metric_list}

    def update(self, y_pred, y_true):
        for metric in self.metric_list:
            self.metrics[metric].update(METRIC_DICT[metric](y_true, y_pred))

    def compute_metrics(self):
        for metric in self.metric_list:
            self.metrics[metric] = METRIC_DICT[metric](self.y_true, self.y_pred)

    def format_metrics(self):
        result = ""
        for metric in self.metric_list:
            result += "{}: {:.4f} | ".format(metric.upper(), self.metrics[metric].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result

    def to_dict(self):
        return {
            metric: self.metrics[metric].avg for metric in self.metric_list
        }

    def __repr__(self):
        return self.metrics[self.metric_list[0]].avg
    
    def __str__(self):
        return self.format_metrics()
