import math
from typing import Union, cast

import numpy as np
import pandas as pd


def shuffle(dataset: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    if type(dataset) == pd.DataFrame:
        return dataset.sample(frac=1)

    nd_dataset = cast(np.ndarray, dataset)

    index = np.arange(len(nd_dataset))
    np.random.shuffle(index)

    return nd_dataset[index]


def train_test_split(dataset: Union[pd.DataFrame, np.ndarray], train_size: float = 0.7, shuffle_dataset: bool = True) -> tuple:
    if shuffle_dataset:
        dataset = shuffle(dataset)

    train_size = train_size if train_size <= 1 else train_size / 100

    pivot = round(len(dataset) * train_size)

    if type(dataset) == pd.DataFrame:
        train, test = dataset.iloc[: pivot], dataset.iloc[pivot:]
    else:
        train, test = dataset[: pivot], dataset[pivot:]

    return train, test


def split_k_folds(dataset: np.ndarray, k: int = 5, shuffle_dataset: bool = True) -> list:
    k_folds = []

    size = math.floor(len(dataset) / k)

    if type(dataset) == pd.DataFrame:
        for i in range(k):
            if i == k - 1:
                k_folds.append(dataset.iloc[size*i:])
            else:
                k_folds.append(dataset.iloc[size*i: size*(i+1)])

    else:
        for i in range(k):
            if i == k - 1:
                k_folds.append(dataset[size*i:])
            else:
                k_folds.append(dataset[size*i: size*(i+1)])

    return k_folds
