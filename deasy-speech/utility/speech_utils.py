

import os

import numpy as np
from skimage.measure import block_reduce
from tqdm import tqdm


def average_pooling(data, pooling_size):
    return block_reduce(data, (pooling_size, 1), np.mean)


def load_audio_features(datadir, filenames=None, list_items=None):
    X = []

    if filenames is None:
        filenames = os.listdir(datadir)

    if list_items is None:
        to_load = filenames
    else:
        to_load = list_items

    for name in tqdm(to_load):
        filepath = os.path.join(datadir, '{0}.{1}')

        # Check preloaded
        preloaded_path = filepath.format(name, 'npy')
        if os.path.isfile(preloaded_path):
            item_features = np.load(preloaded_path)
        else:
            mfcc_path = filepath.format(name, 'txt')
            with open(mfcc_path) as f:
                # MFCC coefficients are a matrix
                item_features = []
                for idx, line in enumerate(f):
                    values = line.strip().split(" ")
                    # Row is a single MFCC feature
                    features_row = []
                    for v in values:
                        features_row.append(float(v))
                    item_features.append(features_row)

            item_features = np.array(item_features)

            # Save for fast load
            np.save(preloaded_path, item_features)

        X.append(item_features)

    # [# audio_files, # mfcc, values]
    return X


def parse_audio_features(data, pooling_sizes=None, remove_energy=False):
    mean_pooling = True if pooling_sizes else False

    if remove_energy:
        data = data[:, 1:]

    if mean_pooling:
        for pooling_size in pooling_sizes:
            data = average_pooling(data, pooling_size)

    return data


def normalize_speaker_audio(data):
    # [frames, #mfccs]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    norm_data = (data - mean[np.newaxis, :]) / std[np.newaxis, :]
    return norm_data
