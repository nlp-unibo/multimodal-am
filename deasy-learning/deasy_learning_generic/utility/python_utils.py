

import collections
import copy
from collections import OrderedDict
from itertools import tee, product

import numpy as np
import pandas as pd
import os
import shutil

import inspect
from deasy_learning_generic.utility.log_utils import Logger


def flatten(d, parent_key='', sep='_'):
    """
    Flattens dictionary
    @see https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys

    :param d: nested dictionary to flatten
    :param parent_key: handle for parent keys path
    :param sep: separator between nested keys
    :return: flattened dict
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return OrderedDict(items)


def _merge(a, b, path=None, overwrite_conflict=True):
    if path is None:
        path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge(a[key], b[key], path + [str(key)])
            # elif a[key] == b[key]:
            #     pass    # same leaf value
            else:
                if overwrite_conflict:
                    a[key] = b[key]
                else:
                    pass
        else:
            a[key] = b[key]
    return a


def merge(a, b, path=None, overwrite_conflict=True):
    """
    merges b into a
    @see https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge

    :param a: dictionary
    :param b: dictionary
    :param path: handle used during dictionary navigation
    :param overwrite_conflict: whether to overwrite a value with b value for a given key in common
    :return: merged dictionary
    """

    # Copying a in order to avoid data modification
    # c = copy.deepcopy(a)
    # return _merge(c, b, path, overwrite_conflict)
    merged = {**a, **b}
    return merged


# s -> (s0,s1), (s1,s2), (s2, s3), ...
def pairwise(iterable):
    """
    Iterates over adjacent couples within given iterable

    :param iterable:
    :return: list of adjacent couples
    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def flatten_nested_array(x):
    """
    Recursively flattens nested array

    :param x: input array to flatten
    :return: flattened array

    Example:

        Input: [1, [2], [2, 3], [[1], 2]]

        Output: [1, 2, 2, 3, 1, 2]

    """

    return [item for seq in x for item in seq]


def parse_value_for_dataframe_condition(value, dataframe):
    """
    Simple trick for filtering dataframe when column value is a list.
    See: https://stackoverflow.com/questions/35255196/pandas-dataframe-condition-when-value-in-cell-is-a-list

    :param value: column value of a given row
    :param dataframe: input dataframe
    :return: pandas.Series containing replicas of the given value if the latter is a list, otherwise the value itself
    """

    if type(value) is list:
        return pd.Series([value], index=dataframe.index)

    return value


def skip_null_conditions(df_series, value):
    """
    Simple trick for skipping None == None condition when filtering a DataFrame instance

    :param df_series: DataFrame column values
    :param value: current value to look for
    :return: value condition is replaced on whether the series has None values or not
    """

    if value is None or value is np.nan:
        transformed_series = df_series.isnull()
        return transformed_series == True

    return df_series == value


def extract_instance_args(instance, kwargs):
    instance_args = {key: value for key, value in kwargs.items() if hasattr(instance, key)}
    remaining_args = {key: value for key, value in kwargs.items() if key not in instance_args}
    return instance_args, remaining_args


def extract_method_args(method, full_args):
    signature_args = list(inspect.signature(method).parameters.keys())
    retrieved_method_args = {key: value for key, value in full_args.items() if key in signature_args}
    remaining_args = {key: value for key, value in full_args.items() if key not in retrieved_method_args}

    return retrieved_method_args, remaining_args


def get_top_n_values2d(arr, n):
    indexes = np.argsort(arr, axis=1)[:, -n:][::-1]
    base_indexes = np.ones_like(indexes) * np.arange(arr.shape[0]).reshape(-1, 1)
    return arr[base_indexes, indexes]


def get_gridsearch_parameters(params_dict):
    """
    Builds parameters combinations

    :param params_dict: dictionary that has parameter names as keys and the list of possible values as values
    (see model_gridsearch.json for more information)
    :return: list of dictionaries, each describing a parameters combination
    """

    params_combinations = []

    keys = sorted(params_dict)
    comb_tuples = product(*(params_dict[key] for key in keys))

    for comb_tuple in comb_tuples:
        instance_params = {dict_key: comb_item for dict_key, comb_item in zip(keys, comb_tuple)}
        params_combinations.append(instance_params)

    return params_combinations


def list_to_set_keep_order(sequence):
    seen = set()
    seen_add = seen.add
    return [element for element in sequence if not (element in seen or seen_add(element))]


def clear_folder(folder):
    assert os.path.isdir(folder), f'{folder} is not a directory.'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            Logger.get_logger(__name__).info(f'Failed to delete {file_path}. Reason: {e}')