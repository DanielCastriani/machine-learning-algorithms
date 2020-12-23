import numpy as np
import pandas as pd
from utils.preprocessing import to_ndarray, parse_error_metrics_list


def confusion_matrix(true, pred, proba=False):
    labels = np.unique(true)
    if len(labels) <= 1:
        raise Exception("expected two or more classes")

    true = to_ndarray(true)
    pred = to_ndarray(pred)

    conf = np.zeros(shape=(len(labels), len(labels)))

    for i in range(len(true)):
        t = true[i]
        p = pred[i]
        conf[t][p] += 1

    return conf / len(true) if proba else conf


def accuracy(true, pred):
    conf = confusion_matrix(true, pred)
    acc = 0

    for i in range(len(conf)):
        acc += conf[i][i]
    return acc / len(true)


def mean_absolute_error(y_pred, y_true):
    y_true, y_pred = parse_error_metrics_list(y_true, y_pred)
    err = np.abs(y_true - y_pred)
    return np.sum(err) / len(y_pred)


def mean_squared_error(y_pred, y_true):
    y_true, y_pred = parse_error_metrics_list(y_true, y_pred)
    err = (y_true - y_pred) ** 2
    return np.sum(err) / len(y_pred)


def root_mean_squared_error(y_pred, y_true):
    y_true, y_pred = parse_error_metrics_list(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_pred, y_true))
