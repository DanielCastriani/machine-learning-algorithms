import pandas as pd
import numpy as np
from typing import Union


class KNNClassifier:
    k: int
    x: np.array
    y: np.array
    labels: np.array

    def __init__(self, k=3):
        self.k = k

    def fit(self, x: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, np.array]):

        if type(x) == pd.DataFrame:
            x = np.array(x.values.tolist())

        if type(y) == pd.DataFrame or type(y) == pd.Series:
            y = np.array(y.values.tolist())

        self.x = x
        self.y = y
        self.labels = np.unique(self.y)

    def count_labels(self, labels: np.array):
        labels_dict = dict()

        for label in self.labels:
            labels_dict[label] = 0

        for label in labels:
            labels_dict[label] += 1

        labels_dict = list(labels_dict.items())
        labels_dict = sorted(
            labels_dict, key=lambda item: item[1], reverse=True)

        labels_dict = [(item[0], item[1] / self.k) for item in labels_dict]

        return labels_dict

    def euclidian_distance(self, sample):
        diff = sample - self.x
        elms_pow = diff ** 2
        elms_sum = np.sum(elms_pow, axis=1)
        return np.sqrt(elms_sum)

    def predict_one(self, sample:  np.array):
        distances = self.euclidian_distance(sample)
        index = distances = np.argsort(distances)
        k_best_index = index[:self.k]
        k_best = self.y[k_best_index]

        return self.count_labels(k_best)

    def predict(self, values: Union[pd.DataFrame, np.array]):
        if type(values) == pd.DataFrame:
            values = values.values.tolist()

        return np.array([self.predict_one(val)[0][0] for val in values])

    def predict_proba(self, values: Union[pd.DataFrame, np.array]):
        if type(values) == pd.DataFrame:
            values = values.values.tolist()

        return np.array([self.predict_one(val)[0] for val in values])
