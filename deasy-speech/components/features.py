from collections import OrderedDict

import numpy as np
import tensorflow as tf

from components.examples import ArgAAAIExample, MArgExample, UsElecExample
from deasy_learning_generic.registry import ComponentFlag, ProjectRegistry
from deasy_learning_tf.features import create_int_feature, create_float_feature
from deasy_learning_tf.nlp.features import TFBaseTextFeatures


class ArgAAAIFeatures(TFBaseTextFeatures):

    def __init__(self, label_id, speech_ids=None, speech_mfccs=None, speech_audio=None):
        super(ArgAAAIFeatures, self).__init__()
        self.speech_ids = speech_ids
        self.speech_mfccs = speech_mfccs
        self.speech_audio = speech_audio
        self.label_id = label_id

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']
        data_mode = converter_info['data_mode']
        use_audio_features = converter_info['use_audio_features']

        mappings = dict()

        if data_mode != 'audio_only':
            mappings['speech_ids'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)

        if data_mode != 'text_only':
            if use_audio_features:
                mappings['speech_mfccs'] = tf.io.FixedLenFeature([max_frame_length * max_frame_features_length],
                                                                 tf.float32)
            else:
                mappings['speech_audio'] = tf.io.FixedLenFeature([max_frame_features_length],
                                                                 tf.float32)

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        converter_info=converter_info,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        data_mode = converter_info['data_mode']
        use_audio_features = converter_info['use_audio_features']

        features = OrderedDict()

        if data_mode != 'audio_only':
            assert feature.speech_ids is not None
            features['speech_ids'] = create_int_feature(feature.speech_ids)

        if data_mode != 'text_only':
            if use_audio_features:
                features['speech_mfccs'] = create_float_feature(feature.speech_mfccs.tolist())
            else:
                features['speech_audio'] = create_float_feature(feature.speech_audio.tolist())

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_info=converter_info)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {}
            for key in ['speech_ids', 'speech_mfccs', 'speech_audio']:
                if key in record:
                    x[key] = record[key]

            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        label_id = cls._convert_labels(example_label=example.label,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        # Speech text if any
        speech_ids = None
        if converter_info['data_mode'] != 'audio_only':
            assert ComponentFlag.TOKENIZER in converter_info['children']
            tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]
            speech_tokens = tokenizer.tokenize(example.speech_text)
            speech_ids = tokenizer.convert_tokens_to_ids(speech_tokens)

        # Speech audio if any
        speech_mfccs = None
        speech_audio = None
        if converter_info['data_mode'] != 'text_only':
            speech_mfccs = example.speech_mfccs
            speech_audio = example.speech_audio_data

        return speech_ids, speech_mfccs, speech_audio, label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        if not isinstance(example, ArgAAAIExample):
            raise AttributeError('Expected ArgAAAIExample instance, got: {}'.format(type(example)))

        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']

        speech_ids, speech_mfccs, speech_audio, \
        label_id = ArgAAAIFeatures.convert_example(example=example, label_list=label_list,
                                                   has_labels=has_labels,
                                                   converter_info=converter_info)

        # Padding
        if speech_ids is not None:
            speech_ids += [0] * (max_speech_length - len(speech_ids))
            speech_ids = speech_ids[:max_speech_length]

        if speech_mfccs is not None:
            speech_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(speech_mfccs)))
            if len(speech_mfccs_pad):
                speech_mfccs = np.concatenate((speech_mfccs, speech_mfccs_pad), axis=0)
            speech_mfccs = speech_mfccs[:max_frame_length]
            speech_mfccs = speech_mfccs.ravel()

        if speech_ids is not None:
            assert len(speech_ids) == max_speech_length
        if speech_mfccs is not None:
            assert len(speech_mfccs) == max_frame_length * max_frame_features_length
        if speech_audio is not None:
            assert len(speech_audio) == max_frame_features_length

        feature = cls(speech_ids=speech_ids, speech_mfccs=speech_mfccs,
                      speech_audio=speech_audio, label_id=label_id)
        return feature


class ArgAAAIBertFeatures(ArgAAAIFeatures):

    def __init__(self, speech_attention_mask=None, **kwargs):
        super(ArgAAAIBertFeatures, self).__init__(**kwargs)
        self.speech_attention_mask = speech_attention_mask

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        mappings = super(ArgAAAIBertFeatures, cls).get_mappings(converter_info=converter_info,
                                                                has_labels=has_labels)

        max_speech_length = converter_info['max_speech_length']
        data_mode = converter_info['data_mode']

        if data_mode != 'audio_only':
            mappings['speech_attention_mask'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        features = super(ArgAAAIBertFeatures, cls).get_feature_records(feature=feature,
                                                                       converter_info=converter_info)

        data_mode = converter_info['data_mode']

        if data_mode != 'audio_only':
            assert feature.speech_attention_mask is not None
            features['speech_attention_mask'] = create_int_feature(feature.speech_attention_mask)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {}
            for key in ['speech_ids', 'speech_attention_mask', 'speech_mfccs', 'speech_audio']:
                if key in record:
                    x[key] = record[key]

            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        label_id = cls._convert_labels(example_label=example.label,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        # Speech text if any
        speech_ids, speech_attention_mask = None, None
        if converter_info['data_mode'] != 'audio_only':
            assert ComponentFlag.TOKENIZER in converter_info['children']
            tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]
            speech_info = tokenizer.tokenize(example.speech_text)
            speech_ids, speech_attention_mask = speech_info['input_ids'], speech_info['attention_mask']

        # Speech audio if any
        speech_mfccs = None
        speech_audio = None
        if converter_info['data_mode'] != 'text_only':
            speech_mfccs = example.speech_mfccs
            speech_audio = example.speech_audio_data

        return speech_ids, speech_attention_mask, speech_mfccs, speech_audio, label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        if not isinstance(example, ArgAAAIExample):
            raise AttributeError('Expected ArgAAAIExample instance, got: {}'.format(type(example)))

        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']

        speech_ids, speech_attention_mask, speech_mfccs, speech_audio, \
        label_id = ArgAAAIBertFeatures.convert_example(example=example, label_list=label_list,
                                                       has_labels=has_labels,
                                                       converter_info=converter_info)

        # Padding
        if speech_ids is not None:
            speech_ids += [0] * (max_speech_length - len(speech_ids))
            speech_ids = speech_ids[:max_speech_length]

            speech_attention_mask += [0] * (max_speech_length - len(speech_attention_mask))
            speech_attention_mask = speech_attention_mask[:max_speech_length]

        if speech_mfccs is not None:
            speech_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(speech_mfccs)))
            if len(speech_mfccs_pad):
                speech_mfccs = np.concatenate((speech_mfccs, speech_mfccs_pad), axis=0)
            speech_mfccs = speech_mfccs[:max_frame_length, :]
            speech_mfccs = speech_mfccs.ravel()

        if speech_ids is not None:
            assert len(speech_ids) == max_speech_length
            assert len(speech_attention_mask) == max_speech_length
        if speech_mfccs is not None:
            assert len(speech_mfccs) == max_frame_length * max_frame_features_length
        if speech_audio is not None:
            assert len(speech_audio) == max_frame_features_length

        feature = cls(speech_ids=speech_ids, speech_attention_mask=speech_attention_mask,
                      speech_mfccs=speech_mfccs, speech_audio=speech_audio, label_id=label_id)
        return feature


def register_arg_aaai_feature_components():
    ProjectRegistry.register_component(class_type=ArgAAAIFeatures,
                                       flag=ComponentFlag.FEATURE,
                                       namespace='arg_aaai',
                                       framework='tf')

    ProjectRegistry.register_component(class_type=ArgAAAIBertFeatures,
                                       flag=ComponentFlag.FEATURE,
                                       namespace='arg_aaai',
                                       tags=['bert'],
                                       framework='tf')


class MArgFeatures(TFBaseTextFeatures):

    def __init__(self, label_id, text_a_ids=None, text_b_ids=None,
                 audio_a_mfccs=None, audio_b_mfccs=None,
                 audio_a_data=None, audio_b_data=None):
        super(MArgFeatures, self).__init__()
        self.text_a_ids = text_a_ids
        self.text_b_ids = text_b_ids
        self.audio_a_mfccs = audio_a_mfccs
        self.audio_b_mfccs = audio_b_mfccs
        self.audio_a_data = audio_a_data
        self.audio_b_data = audio_b_data

        self.label_id = label_id

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']
        data_mode = converter_info['data_mode']
        use_audio_features = converter_info['use_audio_features']

        mappings = dict()

        if data_mode != 'audio_only':
            mappings['text_a_ids'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)
            mappings['text_b_ids'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)

        if data_mode != 'text_only':
            if use_audio_features:
                mappings['audio_a_mfccs'] = tf.io.FixedLenFeature([max_frame_length * max_frame_features_length],
                                                                  tf.float32)
                mappings['audio_b_mfccs'] = tf.io.FixedLenFeature([max_frame_length * max_frame_features_length],
                                                                  tf.float32)
            else:
                mappings['audio_a_data'] = tf.io.FixedLenFeature([max_frame_features_length],
                                                                 tf.float32)
                mappings['audio_b_data'] = tf.io.FixedLenFeature([max_frame_features_length],
                                                                 tf.float32)

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        converter_info=converter_info,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        data_mode = converter_info['data_mode']
        use_audio_features = converter_info['use_audio_features']

        features = OrderedDict()

        if data_mode != 'audio_only':
            assert feature.text_a_ids is not None
            assert feature.text_b_ids is not None
            features['text_a_ids'] = create_int_feature(feature.text_a_ids)
            features['text_b_ids'] = create_int_feature(feature.text_b_ids)

        if data_mode != 'text_only':
            if use_audio_features:
                features['audio_a_mfccs'] = create_float_feature(feature.audio_a_mfccs.tolist())
                features['audio_b_mfccs'] = create_float_feature(feature.audio_b_mfccs.tolist())
            else:
                features['audio_a_data'] = create_float_feature(feature.audio_a_data.tolist())
                features['audio_b_data'] = create_float_feature(feature.audio_b_data.tolist())

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_info=converter_info)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {}
            for key in ['text_a_ids', 'text_b_ids', 'audio_a_mfccs', 'audio_b_mfccs',
                        'audio_a_data', 'audio_b_data']:
                if key in record:
                    x[key] = record[key]

            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        label_id = cls._convert_labels(example_label=example.label,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        # Speech text if any
        text_a_ids, text_b_ids = None, None
        if converter_info['data_mode'] != 'audio_only':
            assert ComponentFlag.TOKENIZER in converter_info['children']
            tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]

            text_a_tokens = tokenizer.tokenize(example.text_a)
            text_a_ids = tokenizer.convert_tokens_to_ids(text_a_tokens)

            text_b_tokens = tokenizer.tokenize(example.text_b)
            text_b_ids = tokenizer.convert_tokens_to_ids(text_b_tokens)

        # Speech audio if any
        audio_a_mfccs = None
        audio_a_data = None
        audio_b_mfccs = None
        audio_b_data = None
        if converter_info['data_mode'] != 'text_only':
            audio_a_mfccs = example.audio_a
            audio_a_data = example.audio_a_data
            audio_b_mfccs = example.audio_b
            audio_b_data = example.audio_b_data

        return text_a_ids, text_b_ids, audio_a_mfccs, audio_a_data, audio_b_mfccs, audio_b_data, label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        if not isinstance(example, MArgExample):
            raise AttributeError('Expected MArgExample instance, got: {}'.format(type(example)))

        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']

        text_a_ids, text_b_ids, audio_a_mfccs, audio_a_data, audio_b_mfccs, audio_b_data, \
        label_id = cls.convert_example(example=example, label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        # Padding
        if text_a_ids is not None:
            text_a_ids += [0] * (max_speech_length - len(text_a_ids))
            text_a_ids = text_a_ids[:max_speech_length]

            text_b_ids += [0] * (max_speech_length - len(text_b_ids))
            text_b_ids = text_b_ids[:max_speech_length]

        if audio_a_mfccs is not None:
            audio_a_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(audio_a_mfccs)))
            if len(audio_a_mfccs_pad):
                audio_a_mfccs = np.concatenate((audio_a_mfccs, audio_a_mfccs_pad), axis=0)
            audio_a_mfccs = audio_a_mfccs[:max_frame_length, :]
            audio_a_mfccs = audio_a_mfccs.ravel()

            audio_b_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(audio_b_mfccs)))
            if len(audio_b_mfccs_pad):
                audio_b_mfccs = np.concatenate((audio_b_mfccs, audio_b_mfccs_pad), axis=0)
            audio_b_mfccs = audio_b_mfccs[:max_frame_length, :]
            audio_b_mfccs = audio_b_mfccs.ravel()

        if text_a_ids is not None:
            assert len(text_a_ids) == max_speech_length
            assert len(text_b_ids) == max_speech_length

        if audio_a_mfccs is not None:
            assert len(audio_a_mfccs) == max_frame_length * max_frame_features_length
            assert len(audio_b_mfccs) == max_frame_length * max_frame_features_length

        if audio_a_data is not None:
            assert len(audio_a_data) == max_frame_features_length
            assert len(audio_b_data) == max_frame_features_length

        feature = cls(text_a_ids=text_a_ids, text_b_ids=text_b_ids,
                      audio_a_mfccs=audio_a_mfccs, audio_a_data=audio_a_data,
                      audio_b_mfccs=audio_b_mfccs, audio_b_data=audio_b_data,
                      label_id=label_id)
        return feature


class MArgBertFeatures(MArgFeatures):

    def __init__(self, text_a_attention_mask=None, text_b_attention_mask=None, **kwargs):
        super(MArgBertFeatures, self).__init__(**kwargs)
        self.text_a_attention_mask = text_a_attention_mask
        self.text_b_attention_mask = text_b_attention_mask

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        mappings = super(MArgBertFeatures, cls).get_mappings(converter_info=converter_info,
                                                             has_labels=has_labels)

        max_speech_length = converter_info['max_speech_length']
        data_mode = converter_info['data_mode']

        if data_mode != 'audio_only':
            mappings['text_a_attention_mask'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)
            mappings['text_b_attention_mask'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        features = super(MArgBertFeatures, cls).get_feature_records(feature=feature,
                                                                    converter_info=converter_info)

        data_mode = converter_info['data_mode']

        if data_mode != 'audio_only':
            features['text_a_attention_mask'] = create_int_feature(feature.text_a_attention_mask)

            features['text_b_attention_mask'] = create_int_feature(feature.text_b_attention_mask)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {}
            for key in ['text_a_ids', 'text_a_attention_mask',
                        'text_b_ids', 'text_b_attention_mask',
                        'audio_a_mfccs', 'audio_b_mfccs',
                        'audio_a_data', 'audio_b_data']:
                if key in record:
                    x[key] = record[key]

            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        label_id = cls._convert_labels(example_label=example.label,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        # Speech text if any
        text_a_ids, text_a_attention_mask = None, None
        text_b_ids, text_b_attention_mask = None, None
        if converter_info['data_mode'] != 'audio_only':
            assert ComponentFlag.TOKENIZER in converter_info['children']
            tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]

            text_a_info = tokenizer.tokenize(example.text_a)
            text_a_ids, text_a_attention_mask = text_a_info['input_ids'], text_a_info['attention_mask']

            text_b_info = tokenizer.tokenize(example.text_b)
            text_b_ids, text_b_attention_mask = text_b_info['input_ids'], text_b_info['attention_mask']

        # Speech audio if any
        audio_a_mfccs = None
        audio_a_data = None
        audio_b_mfccs = None
        audio_b_data = None
        if converter_info['data_mode'] != 'text_only':
            audio_a_mfccs = example.audio_a
            audio_a_data = example.audio_a_data
            audio_b_mfccs = example.audio_b
            audio_b_data = example.audio_b_data

        return text_a_ids, text_a_attention_mask, \
               text_b_ids, text_b_attention_mask, \
               audio_a_mfccs, audio_a_data, audio_b_mfccs, audio_b_data, label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        if not isinstance(example, MArgExample):
            raise AttributeError('Expected MArgExample instance, got: {}'.format(type(example)))

        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']

        text_a_ids, text_a_attention_mask, \
        text_b_ids, text_b_attention_mask, \
        audio_a_mfccs, audio_a_data, audio_b_mfccs, audio_b_data, \
        label_id = MArgBertFeatures.convert_example(example=example, label_list=label_list,
                                                    has_labels=has_labels,
                                                    converter_info=converter_info)

        # Padding
        if text_a_ids is not None:
            text_a_ids += [0] * (max_speech_length - len(text_a_ids))
            text_a_ids = text_a_ids[:max_speech_length]

            text_a_attention_mask += [0] * (max_speech_length - len(text_a_attention_mask))
            text_a_attention_mask = text_a_attention_mask[:max_speech_length]

            text_b_ids += [0] * (max_speech_length - len(text_b_ids))
            text_b_ids = text_b_ids[:max_speech_length]

            text_b_attention_mask += [0] * (max_speech_length - len(text_b_attention_mask))
            text_b_attention_mask = text_b_attention_mask[:max_speech_length]

        if audio_a_mfccs is not None:
            audio_a_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(audio_a_mfccs)))
            audio_a_mfccs_pad = np.reshape(audio_a_mfccs_pad, [-1, max_frame_features_length])
            if len(audio_a_mfccs_pad):
                audio_a_mfccs = np.concatenate((audio_a_mfccs, audio_a_mfccs_pad), axis=0)
            audio_a_mfccs = audio_a_mfccs[:max_frame_length, :]
            audio_a_mfccs = audio_a_mfccs.ravel()

            audio_b_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(audio_b_mfccs)))
            audio_b_mfccs_pad = np.reshape(audio_b_mfccs_pad, [-1, max_frame_features_length])
            if len(audio_b_mfccs_pad):
                audio_b_mfccs = np.concatenate((audio_b_mfccs, audio_b_mfccs_pad), axis=0)
            audio_b_mfccs = audio_b_mfccs[:max_frame_length, :]
            audio_b_mfccs = audio_b_mfccs.ravel()

        if text_a_ids is not None:
            assert len(text_a_ids) == max_speech_length
            assert len(text_a_attention_mask) == max_speech_length
            assert len(text_b_ids) == max_speech_length
            assert len(text_b_attention_mask) == max_speech_length

        if audio_a_mfccs is not None:
            assert len(audio_a_mfccs) == max_frame_length * max_frame_features_length
            assert len(audio_b_mfccs) == max_frame_length * max_frame_features_length

        if audio_a_data is not None:
            assert len(audio_a_data) == max_frame_features_length
            assert len(audio_b_data) == max_frame_features_length

        feature = cls(text_a_ids=text_a_ids, text_a_attention_mask=text_a_attention_mask,
                      text_b_ids=text_b_ids, text_b_attention_mask=text_b_attention_mask,
                      audio_a_mfccs=audio_a_mfccs, audio_b_mfccs=audio_b_mfccs,
                      audio_a_data=audio_a_data, audio_b_data=audio_b_data,
                      label_id=label_id)
        return feature


def register_marg_feature_components():
    ProjectRegistry.register_component(class_type=MArgFeatures,
                                       flag=ComponentFlag.FEATURE,
                                       namespace='m-arg',
                                       framework='tf')

    ProjectRegistry.register_component(class_type=MArgBertFeatures,
                                       flag=ComponentFlag.FEATURE,
                                       namespace='m-arg',
                                       tags=['bert'],
                                       framework='tf')


class UsElecFeatures(TFBaseTextFeatures):

    def __init__(self, label_id, speech_ids=None, speech_mfccs=None, speech_audio=None):
        super(UsElecFeatures, self).__init__()
        self.speech_ids = speech_ids
        self.speech_mfccs = speech_mfccs
        self.speech_audio = speech_audio
        self.label_id = label_id

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']
        data_mode = converter_info['data_mode']
        use_audio_features = converter_info['use_audio_features']

        mappings = dict()

        if data_mode != 'audio_only':
            mappings['speech_ids'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)

        if data_mode != 'text_only':
            if use_audio_features:
                mappings['speech_mfccs'] = tf.io.FixedLenFeature([max_frame_length * max_frame_features_length],
                                                                 tf.float32)
            else:
                mappings['speech_audio'] = tf.io.FixedLenFeature([max_frame_features_length],
                                                                 tf.float32)

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        converter_info=converter_info,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        data_mode = converter_info['data_mode']
        use_audio_features = converter_info['use_audio_features']

        features = OrderedDict()

        if data_mode != 'audio_only':
            assert feature.speech_ids is not None
            features['speech_ids'] = create_int_feature(feature.speech_ids)

        if data_mode != 'text_only':
            if use_audio_features:
                features['speech_mfccs'] = create_float_feature(feature.speech_mfccs.tolist())
            else:
                features['speech_audio'] = create_float_feature(feature.speech_audio.tolist())

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_info=converter_info)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {}
            for key in ['speech_ids', 'speech_mfccs', 'speech_audio']:
                if key in record:
                    x[key] = record[key]

            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        label_id = cls._convert_labels(example_label=example.label,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        # Speech text if any
        speech_ids = None
        if converter_info['data_mode'] != 'audio_only':
            assert ComponentFlag.TOKENIZER in converter_info['children']
            tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]
            speech_tokens = tokenizer.tokenize(example.speech_text)
            speech_ids = tokenizer.convert_tokens_to_ids(speech_tokens)

        # Speech audio if any
        speech_mfccs = None
        speech_audio = None
        if converter_info['data_mode'] != 'text_only':
            speech_mfccs = example.speech_mfccs
            speech_audio = example.speech_audio_data

        return speech_ids, speech_mfccs, speech_audio, label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        if not isinstance(example, UsElecExample):
            raise AttributeError('Expected UsElecExample instance, got: {}'.format(type(example)))

        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']

        speech_ids, speech_mfccs, speech_audio, \
        label_id = UsElecFeatures.convert_example(example=example, label_list=label_list,
                                                  has_labels=has_labels,
                                                  converter_info=converter_info)

        # Padding
        if speech_ids is not None:
            speech_ids += [0] * (max_speech_length - len(speech_ids))
            speech_ids = speech_ids[:max_speech_length]

        if speech_mfccs is not None:
            speech_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(speech_mfccs)))
            if len(speech_mfccs_pad):
                speech_mfccs = np.concatenate((speech_mfccs, speech_mfccs_pad), axis=0)
            speech_mfccs = speech_mfccs[:max_frame_length]
            speech_mfccs = speech_mfccs.ravel()

        if speech_ids is not None:
            assert len(speech_ids) == max_speech_length
        if speech_mfccs is not None:
            assert len(speech_mfccs) == max_frame_length * max_frame_features_length
        if speech_audio is not None:
            assert len(speech_audio) == max_frame_features_length

        feature = cls(speech_ids=speech_ids, speech_mfccs=speech_mfccs,
                      speech_audio=speech_audio, label_id=label_id)
        return feature


class UsElecBertFeatures(UsElecFeatures):

    def __init__(self, speech_attention_mask=None, **kwargs):
        super(UsElecBertFeatures, self).__init__(**kwargs)
        self.speech_attention_mask = speech_attention_mask

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        mappings = super(UsElecBertFeatures, cls).get_mappings(converter_info=converter_info,
                                                               has_labels=has_labels)

        max_speech_length = converter_info['max_speech_length']
        data_mode = converter_info['data_mode']

        if data_mode != 'audio_only':
            mappings['speech_attention_mask'] = tf.io.FixedLenFeature([max_speech_length], tf.int64)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        features = super(UsElecBertFeatures, cls).get_feature_records(feature=feature,
                                                                      converter_info=converter_info)

        data_mode = converter_info['data_mode']

        if data_mode != 'audio_only':
            assert feature.speech_attention_mask is not None
            features['speech_attention_mask'] = create_int_feature(feature.speech_attention_mask)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {}
            for key in ['speech_ids', 'speech_attention_mask', 'speech_mfccs', 'speech_audio']:
                if key in record:
                    x[key] = record[key]

            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        label_id = cls._convert_labels(example_label=example.label,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_info=converter_info)

        # Speech text if any
        speech_ids, speech_attention_mask = None, None
        if converter_info['data_mode'] != 'audio_only':
            assert ComponentFlag.TOKENIZER in converter_info['children']
            tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]
            speech_info = tokenizer.tokenize(example.speech_text)
            speech_ids, speech_attention_mask = speech_info['input_ids'], speech_info['attention_mask']

        # Speech audio if any
        speech_mfccs = None
        speech_audio = None
        if converter_info['data_mode'] != 'text_only':
            speech_mfccs = example.speech_mfccs
            speech_audio = example.speech_audio_data

        return speech_ids, speech_attention_mask, speech_mfccs, speech_audio, label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        if not isinstance(example, UsElecExample):
            raise AttributeError('Expected UsElecExample instance, got: {}'.format(type(example)))

        max_speech_length = converter_info['max_speech_length']
        max_frame_length = converter_info['max_frame_length']
        max_frame_features_length = converter_info['max_frame_features_length']

        speech_ids, speech_attention_mask, speech_mfccs, speech_audio, \
        label_id = UsElecBertFeatures.convert_example(example=example, label_list=label_list,
                                                      has_labels=has_labels,
                                                      converter_info=converter_info)

        # Padding
        if speech_ids is not None:
            speech_ids += [0] * (max_speech_length - len(speech_ids))
            speech_ids = speech_ids[:max_speech_length]

            speech_attention_mask += [0] * (max_speech_length - len(speech_attention_mask))
            speech_attention_mask = speech_attention_mask[:max_speech_length]

        if speech_mfccs is not None:
            speech_mfccs_pad = np.array([[0] * max_frame_features_length] * (max_frame_length - len(speech_mfccs)))
            if len(speech_mfccs_pad):
                speech_mfccs = np.concatenate((speech_mfccs, speech_mfccs_pad), axis=0)
            speech_mfccs = speech_mfccs[:max_frame_length, :]
            speech_mfccs = speech_mfccs.ravel()

        if speech_ids is not None:
            assert len(speech_ids) == max_speech_length
            assert len(speech_attention_mask) == max_speech_length
        if speech_mfccs is not None:
            assert len(speech_mfccs) == max_frame_length * max_frame_features_length
        if speech_audio is not None:
            assert len(speech_audio) == max_frame_features_length

        feature = cls(speech_ids=speech_ids, speech_attention_mask=speech_attention_mask,
                      speech_mfccs=speech_mfccs, speech_audio=speech_audio, label_id=label_id)
        return feature


def register_us_elec_feature_components():
    ProjectRegistry.register_component(class_type=UsElecFeatures,
                                       flag=ComponentFlag.FEATURE,
                                       namespace='us_elec',
                                       framework='tf')

    ProjectRegistry.register_component(class_type=UsElecBertFeatures,
                                       flag=ComponentFlag.FEATURE,
                                       namespace='us_elec',
                                       tags=['bert'],
                                       framework='tf')


def register_feature_components():
    register_arg_aaai_feature_components()
    register_marg_feature_components()
    register_us_elec_feature_components()
