
import os

import tensorflow as tf
from tqdm import tqdm

from deasy_learning_generic.registry import ComponentFlag
from deasy_learning_generic.converter import BaseConverter
from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.examples import ExampleList
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_tf.data import TFConvertedData
from typing import AnyStr

logger = Logger.get_logger(__name__)


class TFBaseConverter(BaseConverter):

    def __init__(self, checkpoint=None, shuffle_amount=10000, prefetch_amount=100,
                 reshuffle_each_iteration=True, **kwargs):
        super(TFBaseConverter, self).__init__(**kwargs)
        self.checkpoint = checkpoint
        self.shuffle_amount = shuffle_amount
        self.prefetch_amount = prefetch_amount
        self.reshuffle_each_iteration = reshuffle_each_iteration

    def get_serialized_filepath(self, model_path: AnyStr, suffix: DataSplit, save_prefix: AnyStr = None):
        if save_prefix is not None:
            return os.path.join(model_path, '{0}{1}_converter_data'.format(suffix, save_prefix))
        else:
            return os.path.join(model_path, '{0}_converter_data'.format(suffix))

    def _load_data(self, model_path, suffix, component_info=None, save_prefix=None, filepath=None):
        converted_data = TFConvertedData.load(filepath=model_path,
                                              suffix=suffix,
                                              save_prefix=save_prefix)
        converted_data.adjust_to_pipeline(component_info=component_info)

        return converted_data

    def _transform_data(self, data, model_path, suffix, save_prefix=None, component_info=None, filepath=None):
        self.convert_data(examples=data,
                          model_path=model_path,
                          label_list=component_info[ComponentFlag.DATA_LOADER]['labels'],
                          has_labels=component_info[ComponentFlag.DATA_LOADER]['has_labels'],
                          suffix=suffix,
                          save_prefix=save_prefix)

        converted_data = TFConvertedData(filepath=self.output_file,
                                         data_size=len(data),
                                         name_to_features=self.feature_class.get_mappings(
                                             converter_info=self.get_info(),
                                             has_labels=component_info[ComponentFlag.DATA_LOADER]['has_labels']),
                                         selector=self.feature_class.get_dataset_selector(component_info[ComponentFlag.DATA_LOADER]['labels']),
                                         is_training=suffix == DataSplit.TRAIN,
                                         shuffle_amount=self.shuffle_amount,
                                         reshuffle_each_iteration=self.reshuffle_each_iteration,
                                         prefetch_amount=self.prefetch_amount,
                                         batch_size=component_info[ComponentFlag.EVALUATION]['batch_size'])
        converted_data.save(filepath=model_path, suffix=suffix, save_prefix=save_prefix)
        converted_data.adjust_to_pipeline(component_info=component_info)

        return converted_data

    def convert_data(self, examples, model_path, label_list, has_labels=True, save_prefix=None, suffix=DataSplit.TRAIN):
        self.output_file = self.get_serialized_filepath(model_path=model_path,
                                                        save_prefix=save_prefix,
                                                        suffix=suffix)

        assert isinstance(examples, ExampleList)
        if suffix == DataSplit.TRAIN:
            logger.info('Retrieving training set info...this may take a while...')
            self.training_preparation(examples=examples,
                                      label_list=label_list)

        writer = tf.io.TFRecordWriter(self.output_file)

        for ex_index, example in enumerate(tqdm(examples, leave=True, position=0)):
            if self.checkpoint is not None and ex_index % self.checkpoint == 0:
                logger.info('Writing example {0} of {1}'.format(ex_index, len(examples)))

            feature = self.feature_class.from_example(example,
                                                      label_list,
                                                      has_labels=has_labels,
                                                      converter_info=self.get_info())

            features = self.feature_class.get_feature_records(feature, converter_info=self.get_info())

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

        writer.close()
