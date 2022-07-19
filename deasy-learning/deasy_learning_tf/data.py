

from deasy_learning_generic.data import ConvertedData
from deasy_learning_generic.registry import ComponentFlag
from deasy_learning_tf.utility.tensorflow_utils import get_dataset_fn, retrieve_numpy_labels
import numpy as np


class TFConvertedData(ConvertedData):

    def __init__(self, data_size, filepath, name_to_features, selector, batch_size, is_training=False,
                 shuffle_amount=10000, prefetch_amount=100, reshuffle_each_iteration=True,
                 sampling=False, sampler=None):
        super(TFConvertedData, self).__init__()
        self.data_size = data_size
        self.filepath = filepath
        self.name_to_features = name_to_features
        self.selector = selector
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle_amount = shuffle_amount
        self.prefetch_amount = prefetch_amount
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.sampling = sampling
        self.sampler = sampler

        self.iterator = None
        self.training_iterator = None
        self._build_iterators()

    def _build_iterators(self):
        self.iterator = get_dataset_fn(filepath=self.filepath,
                                       name_to_features=self.name_to_features,
                                       selector=self.selector,
                                       is_training=False,
                                       shuffle_amount=self.shuffle_amount,
                                       prefetch_amount=self.prefetch_amount,
                                       reshuffle_each_iteration=self.reshuffle_each_iteration,
                                       batch_size=self.batch_size,
                                       sampling=self.sampling,
                                       sampler=self.sampler)

        if self.is_training:
            self.training_iterator = get_dataset_fn(filepath=self.filepath,
                                                    name_to_features=self.name_to_features,
                                                    selector=self.selector,
                                                    is_training=True,
                                                    shuffle_amount=self.shuffle_amount,
                                                    prefetch_amount=self.prefetch_amount,
                                                    reshuffle_each_iteration=self.reshuffle_each_iteration,
                                                    batch_size=self.batch_size,
                                                    sampling=self.sampling,
                                                    sampler=self.sampler)

    def adjust_to_pipeline(self, component_info):
        self.batch_size = component_info[ComponentFlag.EVALUATION]['batch_size']
        self.steps = int(np.ceil(self.data_size / self.batch_size))
        self._build_iterators()

    def get_data(self):
        return iter(self.iterator())

    def get_labels(self):
        return retrieve_numpy_labels(self.iterator, steps=self.steps)

    def get_training_iterator(self):
        return iter(self.training_iterator())
