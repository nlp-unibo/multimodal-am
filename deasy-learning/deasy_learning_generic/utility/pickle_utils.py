


import dill


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = dill.load(f)

    return data


def save_pickle(filepath, data):
    with open(filepath, 'wb') as f:
        dill.dump(data, f)
