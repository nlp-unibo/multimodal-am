from collections import OrderedDict

from deasy_learning_generic.component import Component
from deasy_learning_generic.nlp.utility import preprocessing_utils
from deasy_learning_generic.utility.python_utils import merge
from deasy_learning_generic.utility.pickle_utils import load_pickle, save_pickle
from deasy_learning_generic.data_loader import DataSplit
import os
from typing import AnyStr, Dict


class DataProcessor(Component):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, filter_names=None, disable_filtering=False, retrieve_label=True, **kwargs):
        super(DataProcessor, self).__init__(**kwargs)
        self.filter_names = filter_names if filter_names is not None else preprocessing_utils.filter_methods
        self.disable_filtering = disable_filtering
        self.retrieve_label = retrieve_label

    def _load_data(self, model_path: AnyStr, suffix: AnyStr, component_info: Dict = None,
                   save_prefix: AnyStr = None, filepath: AnyStr = None):
        return load_pickle(filepath=self.get_serialized_filepath(model_path=model_path,
                                                                 suffix=suffix,
                                                                 save_prefix=save_prefix))

    def _save_data(self, data, model_path: AnyStr, filepath: AnyStr = None,
                   save_prefix: AnyStr = None, suffix: DataSplit = DataSplit.TRAIN):
        save_pickle(filepath=self.get_serialized_filepath(model_path=model_path,
                                                          suffix=suffix,
                                                          save_prefix=save_prefix), data=data)

    def get_serialized_filepath(self, model_path: AnyStr, suffix: DataSplit, save_prefix: AnyStr = None):
        if save_prefix is not None:
            return os.path.join(model_path, '{0}{1}_processor_data'.format(suffix, save_prefix))
        else:
            return os.path.join(model_path, '{0}_processor_data'.format(suffix))

    def apply(self, component_index, data, model_path, suffix, save_prefix=None, pipeline_info=None, filepath=None,
              serialized_component_index=None, is_child=False, save_info=False, show_info=False):
        converted_data, pipeline_info, component_info = super(DataProcessor, self).apply(
            component_index=component_index, data=data,
            model_path=model_path, suffix=suffix,
            save_prefix=save_prefix,
            pipeline_info=pipeline_info, filepath=filepath,
            serialized_component_index=serialized_component_index,
            is_child=is_child, save_info=save_info,
            show_info=show_info)

        if self._should_transform(model_path=model_path, suffix=suffix,
                                  save_prefix=save_prefix, filepath=filepath,
                                  serialized_component_index=serialized_component_index,
                                  component_index=component_index):
            # Add ExampleList state
            pipeline_info = merge(pipeline_info, converted_data.get_added_state())

        return converted_data, pipeline_info, component_info

    def get_filename(self):
        return 'processor_info'

    def _retrieve_default_label(self, labels, item, data_keys):
        if self.retrieve_label:
            label = OrderedDict([(label.name, item[data_keys[label.name]]) for label in labels])
        else:
            label = None

        return label

    def wrap_single_item(self, item):
        raise NotImplementedError()
