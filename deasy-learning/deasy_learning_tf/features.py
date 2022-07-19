

from collections import OrderedDict

import tensorflow as tf

from deasy_learning_generic.features import BaseFeatures
from deasy_learning_generic.utility.log_utils import Logger

logger = Logger.get_logger(__name__)


# Utility

def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


# Features


class TFBaseFeatures(BaseFeatures):

    @classmethod
    def _retrieve_default_label_mappings(cls, mappings, converter_info, has_labels=True):
        label_list = converter_info['label_list']

        if has_labels:
            for label in label_list:
                mappings[label.name] = tf.io.FixedLenFeature([1], tf.int64)

        return mappings

    @classmethod
    def _retrieve_default_label_feature_records(cls, feature, features, converter_info=None):
        if feature.label_id is not None:
            # single label group
            if type(feature.label_id) == OrderedDict:
                for key, value in feature.label_id.items():
                    if type(value) == list:
                        features[key] = create_int_feature(value)
                    else:
                        features[key] = create_int_feature([value])

            # sequence label groups
            else:
                keys = feature.label_id[0].keys()
                for key in keys:
                    key_values = [item[key] for item in feature.label_id]
                    features[key] = create_int_feature(key_values)

        return features


