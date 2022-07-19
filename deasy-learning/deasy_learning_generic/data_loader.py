from abc import ABC
from enum import Enum

from deasy_learning_generic.metrics import MetricManager
from deasy_learning_generic.composable import Composable


class DataSplit(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __str__(self):
        return str(self.value)


class DataLoader(Composable):

    def __init__(self, label_metrics_map=None, metrics=None, **kwargs):
        super(DataLoader, self).__init__(**kwargs)
        self.has_loaded = False

        # TODO: this has to be defined for each potential split
        self.label_metrics_map = label_metrics_map if label_metrics_map is not None else {}
        self.metrics = metrics if metrics is not None else []
        self.metrics_manager = MetricManager(self.label_metrics_map)
        for metric in self.metrics:
            self.metrics_manager.add_metric(metric)
        self.metrics_manager.finalize()

        self.data = {
            DataSplit.TRAIN: None,
            DataSplit.VAL: None,
            DataSplit.TEST: None
        }

    def __getitem__(self, key):
        assert isinstance(key, DataSplit)
        return self.data[key]

    def assign_train_data(self, data):
        self.data[DataSplit.TRAIN] = data

    def assign_val_data(self, data):
        self.data[DataSplit.VAL] = data

    def assign_test_data(self, data):
        self.data[DataSplit.TEST] = data

    def load(self):
        raise NotImplementedError()

    def get_metrics(self):
        return self.metrics_manager

    def build_data_splits(self):
        raise NotImplementedError()

    def get_data_splits(self):
        return self.data[DataSplit.TRAIN], self.data[DataSplit.VAL], self.data[DataSplit.TEST]


class DataHandle(ABC):

    def __init__(self, data, split, data_name, data_keys, labels=None):
        self.data = data
        self.split = split
        self.data_name = data_name
        self.data_keys = data_keys
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError()

    # This should return a valid value for __getitem__
    def belongs_to(self, values, target_values):
        raise NotImplementedError()

    def get_item_value(self, item_idx, value_key):
        raise NotImplementedError()

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def has_labels(self):
        return self.labels is not None

    def get_data_keys(self):
        return self.data_keys

    def get_info(self):
        return {
            'has_labels': self.has_labels(),
            'labels': self.get_labels()
        }
