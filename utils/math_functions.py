import numpy as np


def euclidian_distance(a: np.array, b: np.array):
    diff = a - b
    elms_pow = diff ** 2
    elms_sum = np.sum(elms_pow, axis=1)
    return np.sqrt(elms_sum)
