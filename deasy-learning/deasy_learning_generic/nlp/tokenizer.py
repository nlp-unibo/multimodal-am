

from deasy_learning_generic.component import Component
from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.utility.pickle_utils import load_pickle, save_pickle
import os
from typing import AnyStr, Dict


class BaseTokenizer(Component):

    def __init__(self, embedding_dimension=None,
                 embedding_model_type=None, merge_vocabularies=False, **kwargs):
        super(BaseTokenizer, self).__init__(**kwargs)

        self.embedding_dimension = embedding_dimension
        self.embedding_model_type = embedding_model_type
        self.embedding_model = None
        self.embedding_matrix = None
        self.merge_vocabularies = merge_vocabularies
        self.vocab = None
        self.vocab_size = None

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
            return os.path.join(model_path, '{0}{1}_tokenizer_data'.format(suffix, save_prefix))
        else:
            return os.path.join(model_path, '{0}_tokenizer_data'.format(suffix))

    def _transform_data(self, data, model_path, suffix, save_prefix=None, component_info=None, filepath=None):
        if suffix == DataSplit.TRAIN:
            self.build_vocab(data.get_data())

        self._save_data(data=data, model_path=model_path, suffix=suffix,
                        save_prefix=save_prefix, filepath=filepath)

        return data

    def get_filename(self):
        return 'tokenizer_info'

    def build_vocab(self, data, **kwargs):
        raise NotImplementedError()

    def initialize_with_info(self, info):
        pass

    def tokenize(self, text, remove_special_tokens=False):
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens):
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids):
        raise NotImplementedError()

