from utils.preprocessing import to_ndarray
import pandas as pd
import numpy as np
from utils.math_functions import euclidian_distance
from typing import Union


class KNNClassifier:
    k: int
    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray

    def __init__(self, k=3):
        self.k = k

    def fit(self, x: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]):
        self.x = to_ndarray(x)
        self.y = to_ndarray(y)
        self.labels = np.unique(self.y)

    def count_labels(self, labels: np.ndarray):
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

    def predict_one(self, sample:  np.ndarray):
        distances = euclidian_distance(sample, self.x)
        index = np.argsort(distances)
        k_best_index = index[:self.k]
        k_best = self.y[k_best_index]

        return self.count_labels(k_best)

    def predict(self, values: Union[pd.DataFrame, np.ndarray]):
        if type(values) == pd.DataFrame:
            values = values.values.tolist()

        return np.array([self.predict_one(val)[0][0] for val in values])

    def predict_proba(self, values: Union[pd.DataFrame, np.ndarray]):
        if type(values) == pd.DataFrame:
            values = values.values.tolist()

        return np.array([self.predict_one(val)[0] for val in values])
