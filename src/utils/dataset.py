import os
import re
import math
import numpy as np

from time import time
from typing import Union


def is_int(value: str):
    return re.compile(r'^-{0,1}\d+$').match(value)


def is_float(value: str):
    return re.compile(r'^-{0,1}\d+\.\d+$').match(value)


def load_csv(path: str, sep: str = ',', skip_header: bool = True, parse_np: bool = True, verbose: bool = False) -> Union[list, np.array]:
    start_time = time()

    if not os.path.exists(path):
        raise BaseException("%s don't exists")

    data = []

    with open(path, 'r') as file:
        lines = file.readlines()

        if skip_header:
            lines = lines[1:]

        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            values = line.split(sep)
            values_parse = []
            for value in values:
                try:
                    if is_int(value):
                        values_parse.append(int(value))
                    elif is_float(value):
                        values_parse.append(float(value))
                    else:
                        values_parse.append(value)
                except:
                    values_parse.append(value)

            data.append(values_parse)

    if parse_np:
        data = np.array(data)

    if verbose:
        print(time() - start_time, 's')

    return data


def shuffle(dataset: np.array):
    index = np.arange(len(dataset))
    np.random.shuffle(index)

    return dataset[index]


def train_test_split(dataset: np.array, train_size: float = 0.7, shuffle_dataset: bool = True) -> tuple:
    if shuffle_dataset:
        dataset = shuffle(dataset)

    train_size = train_size if train_size <= 1 else train_size / 100

    pivot = round(len(dataset) * train_size)
    train, test = dataset[: pivot], dataset[pivot:]

    return train, test


def split_k_folds(dataset: np.array, k: int = 5, shuffle_dataset: bool = True) -> list:
    k_folds = []

    size = math.floor(len(dataset) / k)

    for i in range(k):
        if i == k - 1:
            k_folds.append(dataset[size*i:])
        else:
            k_folds.append(dataset[size*i: size*(i+1)])

    return k_folds
