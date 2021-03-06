from typing import Union

import numpy as np
import pandas as pd


def to_ndarray(entries) -> np.ndarray:
    if type(entries) == list:
        return np.array(entries)
    elif type(entries) == pd.Series:
        return np.array(entries.tolist())
    elif type(entries) == pd.DataFrame:
        return np.array(entries.values.tolist())
    else:
        return entries


def parse_error_metrics_list(true, pred):
    return to_ndarray(true), to_ndarray(pred)


def label_encoder(y, labels=None):
    if not labels:
        labels = np.unique(y)

    return [np.where(item == labels)[0][0] for item in y], labels

def one_hot_encoder(data):
    data = to_ndarray(data)

    # Convert categorical values to int: ['A', 'B', 'B', 'C'] -> [0, 1, 1, 2]
    label_encoded, labels = label_encoder(data)
        
    # create a 0 matrix of size n_data x n_categories
    one_hot = np.zeros((len(data), len(labels))) 

    # Encode the values to one hot encode
    # The fist index, access the row, and the second index(label_encoded) access the position corresponding the label(int)
    one_hot[np.arange(len(label_encoded)), label_encoded] = 1
    return one_hot

def min_max_normalize(dataset: Union[pd.DataFrame, np.ndarray], minmax: dict = None):
    """Normalize the data using min max

    Args:
        dataset (pd.DataFrame, np.array): dataset to be normalized
        minmax dict: dictionary of min and max values used to normalize, if is None, the values will be calculated

    Returns:
        (Union[pd.DataFrame, np.array], dict) Normalized dataset, and values used to normalize the dataset

    """
    df = dataset

    parseNP = False
    if type(dataset) != pd.DataFrame:
        df = pd.DataFrame(dataset)
        parseNP = True

    if minmax == None:
        minmax = pd.DataFrame([df.min(), df.max()], index=[
                              'Min', 'Max']).to_dict()

    for col in df.columns:
        norm = minmax[col]
        m = norm['Min']
        M = norm['Max']
        df[col] = (df[col] - m) / (M - m)

    if parseNP:
        normalized_dataset = df.values.tolist()
    else:
        normalized_dataset = df

    return normalized_dataset, minmax
