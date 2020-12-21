import numpy as np
import pandas as pd


def confusion_matrix(true, pred, proba=False):
    labels = np.unique(true)
    if len(labels) <= 1:
        raise Exception("expected two or more classes")

    if type(true) == list:
        true = np.array(true)

    if type(pred) == list:
        pred = np.array(pred)

    if type(true) == pd.Series:
        true = np.array(true.tolist())

    if type(pred) == pd.Series:
        pred = np.array(pred.tolist())

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
