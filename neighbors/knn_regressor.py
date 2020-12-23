from typing import Union
from utils.preprocessing import to_ndarray, parse_error_metrics_list
from utils.math_functions import euclidian_distance

import numpy as np
import pandas as pd


class KNNRegressor:
    """
    Args:
        k: k nearest neighbors
        mode: 'weight' or 'mean'
            * mean: mean of k nearest neighbors
            * weight: weighted mean of k nearest neighbors
    """
    k: int
    mode: str
    x: np.array
    y: np.array
    labels: np.array

    def __init__(self, k: int = 3, mode: str = 'weight'):
        self.k = k
        self.mode = mode

    def fit(self, x: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, np.array]):
        x = to_ndarray(x)
        y = to_ndarray(y)

        self.x = x
        self.y = y
        self.labels = np.unique(self.y)

    def weighted_mean(self, values, distances):
        # Avoid division by zero
        distances[distances == 0] = 1.0e-99
        w = 1/distances
        w = w / np.sum(w)
        return np.sum(values * w)

    def predict_one(self, sample:  np.array):
        distances = euclidian_distance(sample, self.x)
        index = np.argsort(distances)
        k_best_index = index[:self.k]
        k_best = self.y[k_best_index]

        if self.mode == 'mean':
            return np.mean(k_best)

        k_best_distance = distances[k_best_index]

        return self.weighted_mean(k_best, k_best_distance)

    def predict(self, values: Union[pd.DataFrame, np.array]):
        values = to_ndarray(values)

        return np.array([self.predict_one(val) for val in values])
