
from collections import OrderedDict

import tensorflow as tf

from deasy_learning_tf.features import create_int_feature
from deasy_learning_tf.nlp.features import TFBaseTextFeatures
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.implementations.nlp.examples import TextExample
from deasy_learning_generic.registry import ComponentFlag

logger = Logger.get_logger(__name__)


class TFTextFeatures(TFBaseTextFeatures):

    def __init__(self, text_ids, label_id):
        super(TFTextFeatures, self).__init__()
        self.text_ids = text_ids
        self.label_id = label_id

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        max_seq_length = converter_info['max_seq_length']

        mappings = dict()
        mappings['text_ids'] = tf.io.FixedLenFeature([max_seq_length], tf.int64)

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        converter_info=converter_info,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        features = OrderedDict()
        features['text_ids'] = create_int_feature(feature.text_ids)

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_info=converter_info)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {
                'text': record['text_ids'],
            }
            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        assert ComponentFlag.TOKENIZER in converter_info['children']
        tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]

        label_id = cls._convert_labels(example_label=example.label,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        tokens = tokenizer.tokenize(example.text)
        text_ids = tokenizer.convert_tokens_to_ids(tokens)

        return text_ids, label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        if not isinstance(example, TextExample):
            raise AttributeError('Expected TextExample instance, got: {}'.format(type(example)))

        max_seq_length = converter_info['max_seq_length']

        text_ids, label_id = TFTextFeatures.convert_example(example=example, label_list=label_list,
                                                            has_labels=has_labels,
                                                            converter_info=converter_info)

        # Padding
        text_ids += [0] * (max_seq_length - len(text_ids))
        text_ids = text_ids[:max_seq_length]

        assert len(text_ids) == max_seq_length

        feature = TFTextFeatures(text_ids=text_ids, label_id=label_id)
        return feature
