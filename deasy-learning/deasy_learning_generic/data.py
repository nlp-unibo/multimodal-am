from abc import ABC
from deasy_learning_generic.utility.pickle_utils import save_pickle, load_pickle
import os


class ConvertedData(ABC):

    def __init__(self):
        self.additional_data = {}

    def adjust_to_pipeline(self, component_info):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    def get_training_iterator(self):
        raise NotImplementedError()

    def save(self, filepath, suffix, save_prefix=None):
        if save_prefix is not None:
            save_path = os.path.join(filepath, '{0}{1}_data_converter_wrapper.pickle'.format(suffix, save_prefix))
        else:
            save_path = os.path.join(filepath, '{0}_data_converter_wrapper.pickle'.format(suffix))

        # Save only if needed
        if not os.path.isfile(save_path):
            save_pickle(save_path, self)

    @classmethod
    def load(cls, filepath, suffix, save_prefix=None):
        if save_prefix is not None:
            load_path = os.path.join(filepath, '{0}{1}_data_converter_wrapper.pickle'.format(suffix, save_prefix))
        else:
            load_path = os.path.join(filepath, '{0}_data_converter_wrapper.pickle'.format(suffix))

        return load_pickle(load_path)