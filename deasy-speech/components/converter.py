from tqdm import tqdm

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_tf.nlp.converter import TFBaseTextConverter
from deasy_learning_generic.utility.component_utils import NumericInfo

logger = Logger.get_logger(__name__)


class TFIArgAAAIConverter(TFBaseTextConverter):

    def __init__(self, data_mode='text_only', max_frame_limit=None, use_audio_features=True, **kwargs):
        super(TFIArgAAAIConverter, self).__init__(**kwargs)
        self.data_mode = data_mode
        self.max_frame_limit = max_frame_limit if max_frame_limit is not None else 10000000
        self.use_audio_features = use_audio_features

    def training_preparation(self, examples, label_list):
        max_speech_length = NumericInfo(name='max_speech_length')
        max_frame_length = NumericInfo(name='max_frame_length')
        max_frame_features_length = NumericInfo(name='max_frame_features_length')

        for example in tqdm(examples):
            features = self.feature_class.convert_example(example=example,
                                                          label_list=label_list,
                                                          converter_info=self.get_info())

            speech_ids, speech_mfcc, speech_audio, label_id = features

            speech_len = len(speech_ids) if speech_ids is not None else None

            frame_features_length = None
            frame_length = None
            if speech_mfcc is not None:
                frame_length = len(speech_mfcc)
                frame_features_length = speech_mfcc.shape[-1]

            if speech_audio is not None:
                frame_length = None
                frame_features_length = len(speech_audio)

            if speech_len is not None:
                max_speech_length.add(speech_len)

            if frame_length is not None:
                max_frame_length.add(frame_length)

            if frame_features_length is not None:
                max_frame_features_length.add(frame_features_length)

        max_speech_length.summary()
        max_speech_length.show()
        max_frame_length.summary()
        max_frame_length.show()
        max_frame_features_length.summary()
        max_frame_features_length.show()

        self.label_list = label_list
        self.max_speech_length = min(int(max_speech_length.get_info(info_name='op_quantile_0.95')), self.max_tokens_limit)
        self.max_frame_length = min(int(max_frame_length.get_info(info_name='op_quantile_0.95')), self.max_frame_limit)
        self.max_frame_features_length = int(max_frame_features_length.get_info(info_name='op_quantile_0.95'))


class TFArgAAAIBertConverter(TFIArgAAAIConverter):

    def training_preparation(self, examples, label_list):
        max_speech_length = NumericInfo(name='max_speech_length')
        max_frame_length = NumericInfo(name='max_frame_length')
        max_frame_features_length = NumericInfo(name='max_frame_features_length')

        for example in tqdm(examples):
            features = self.feature_class.convert_example(example=example,
                                                          label_list=label_list,
                                                          converter_info=self.get_info())

            speech_ids, speech_attention_mask, speech_mfcc, speech_audio, label_id = features

            speech_len = len(speech_ids) if speech_ids is not None else None

            frame_features_length = None
            frame_length = None
            if speech_mfcc is not None:
                frame_length = len(speech_mfcc)
                frame_features_length = speech_mfcc.shape[-1]

            if speech_audio is not None:
                frame_length = None
                frame_features_length = len(speech_audio)

            if speech_len is not None:
                max_speech_length.add(speech_len)

            if frame_length is not None:
                max_frame_length.add(frame_length)

            if frame_features_length is not None:
                max_frame_features_length.add(frame_features_length)

        max_speech_length.summary()
        max_speech_length.show()
        max_frame_length.summary()
        max_frame_length.show()
        max_frame_features_length.summary()
        max_frame_features_length.show()

        self.label_list = label_list
        self.max_speech_length = min(int(max_speech_length.get_info(info_name='op_quantile_0.95')), self.max_tokens_limit)
        self.max_frame_length = min(int(max_frame_length.get_info(info_name='op_quantile_0.95')), self.max_frame_limit)
        self.max_frame_features_length = int(max_frame_features_length.get_info(info_name='op_quantile_0.95'))


def register_arg_aaai_converter_components():
    ProjectRegistry.register_component(class_type=TFIArgAAAIConverter,
                                       flag=ComponentFlag.CONVERTER,
                                       framework='tf',
                                       namespace='arg_aaai')

    ProjectRegistry.register_component(class_type=TFArgAAAIBertConverter,
                                       flag=ComponentFlag.CONVERTER,
                                       framework='tf',
                                       tags=['bert'],
                                       namespace='arg_aaai')


class TFMArgConverter(TFBaseTextConverter):

    def __init__(self, data_mode='text_only', max_frame_limit=None, use_audio_features=True, **kwargs):
        super(TFMArgConverter, self).__init__(**kwargs)
        self.data_mode = data_mode
        self.max_frame_limit = max_frame_limit if max_frame_limit is not None else 10000000
        self.use_audio_features = use_audio_features

    def training_preparation(self, examples, label_list):
        max_speech_length = NumericInfo(name='max_speech_length')
        max_frame_length = NumericInfo(name='max_frame_length')
        max_frame_features_length = NumericInfo(name='max_frame_features_length')

        for example in tqdm(examples):
            features = self.feature_class.convert_example(example=example,
                                                          label_list=label_list,
                                                          converter_info=self.get_info())

            text_a_ids, text_b_ids, audio_a_mfccs, audio_a_data, audio_b_mfccs, audio_b_data, label_id = features

            speech_length = max(len(text_a_ids), len(text_b_ids)) if text_a_ids is not None else None

            frame_features_length = None
            frame_length = None
            if audio_a_mfccs is not None:
                frame_length = max(len(audio_a_mfccs), len(audio_b_mfccs))
                frame_features_length = audio_a_mfccs.shape[-1]

            if audio_a_data is not None:
                frame_length = None
                frame_features_length = len(audio_a_data)

            if speech_length is not None:
                max_speech_length.add(speech_length)

            if frame_length is not None:
                max_frame_length.add(frame_length)

            if frame_features_length is not None:
                max_frame_features_length.add(frame_features_length)

        max_speech_length.summary()
        max_speech_length.show()
        max_frame_length.summary()
        max_frame_length.show()
        max_frame_features_length.summary()
        max_frame_features_length.show()

        self.label_list = label_list
        self.max_speech_length = min(int(max_speech_length.get_info(info_name='op_quantile_0.95')), self.max_tokens_limit)
        self.max_frame_length = min(int(max_frame_length.get_info(info_name='op_quantile_0.95')), self.max_frame_limit)
        self.max_frame_features_length = int(max_frame_features_length.get_info(info_name='op_quantile_0.95'))


class TFMArgBertConverter(TFMArgConverter):

    def training_preparation(self, examples, label_list):
        max_speech_length = NumericInfo(name='max_speech_length')
        max_frame_length = NumericInfo(name='max_frame_length')
        max_frame_features_length = NumericInfo(name='max_frame_features_length')

        for example in tqdm(examples):
            features = self.feature_class.convert_example(example=example,
                                                          label_list=label_list,
                                                          converter_info=self.get_info())

            text_a_ids, text_a_attention_mask, \
            text_b_ids, text_b_attention_mask, \
            audio_a_mfccs, audio_a_data, audio_b_mfccs, audio_b_data, label_id = features

            speech_length = max(len(text_a_ids), len(text_b_ids)) if text_a_ids is not None else None

            frame_features_length = None
            frame_length = None
            if audio_a_mfccs is not None:
                frame_length = max(len(audio_a_mfccs), len(audio_b_mfccs))
                frame_features_length = audio_a_mfccs.shape[-1]

            if audio_a_data is not None:
                frame_length = None
                frame_features_length = len(audio_a_data)

            if speech_length is not None:
                max_speech_length.add(speech_length)

            if frame_length is not None:
                max_frame_length.add(frame_length)

            if frame_features_length is not None:
                max_frame_features_length.add(frame_features_length)

        max_speech_length.summary()
        max_speech_length.show()
        max_frame_length.summary()
        max_frame_length.show()
        max_frame_features_length.summary()
        max_frame_features_length.show()

        self.label_list = label_list
        self.max_speech_length = min(int(max_speech_length.get_info(info_name='op_quantile_0.95')), self.max_tokens_limit)
        self.max_frame_length = min(int(max_frame_length.get_info(info_name='op_quantile_0.95')), self.max_frame_limit)
        self.max_frame_features_length = int(max_frame_features_length.get_info(info_name='op_quantile_0.95'))


def register_m_arg_converter_components():
    ProjectRegistry.register_component(class_type=TFMArgConverter,
                                       flag=ComponentFlag.CONVERTER,
                                       framework='tf',
                                       namespace='m-arg')

    ProjectRegistry.register_component(class_type=TFMArgBertConverter,
                                       flag=ComponentFlag.CONVERTER,
                                       framework='tf',
                                       tags=['bert'],
                                       namespace='m-arg')


class TFUsElecConverter(TFBaseTextConverter):

    def __init__(self, data_mode='text_only', max_frame_limit=None, use_audio_features=True, **kwargs):
        super(TFUsElecConverter, self).__init__(**kwargs)
        self.data_mode = data_mode
        self.max_frame_limit = max_frame_limit if max_frame_limit is not None else 10000000
        self.use_audio_features = use_audio_features

    def training_preparation(self, examples, label_list):
        max_speech_length = NumericInfo(name='max_speech_length')
        max_frame_length = NumericInfo(name='max_frame_length')
        max_frame_features_length = NumericInfo(name='max_frame_features_length')

        for example in tqdm(examples):
            features = self.feature_class.convert_example(example=example,
                                                          label_list=label_list,
                                                          converter_info=self.get_info())

            speech_ids, speech_mfcc, speech_audio, label_id = features

            speech_len = len(speech_ids) if speech_ids is not None else None

            frame_features_length = None
            frame_length = None
            if speech_mfcc is not None:
                frame_length = len(speech_mfcc)
                frame_features_length = speech_mfcc.shape[-1]

            if speech_audio is not None:
                frame_length = None
                frame_features_length = len(speech_audio)

            if speech_len is not None:
                max_speech_length.add(speech_len)

            if frame_length is not None:
                max_frame_length.add(frame_length)

            if frame_features_length is not None:
                max_frame_features_length.add(frame_features_length)

        max_speech_length.summary()
        max_speech_length.show()
        max_frame_length.summary()
        max_frame_length.show()
        max_frame_features_length.summary()
        max_frame_features_length.show()

        self.label_list = label_list
        self.max_speech_length = min(int(max_speech_length.get_info(info_name='op_quantile_0.95')), self.max_tokens_limit)
        self.max_frame_length = min(int(max_frame_length.get_info(info_name='op_quantile_0.95')), self.max_frame_limit)
        self.max_frame_features_length = int(max_frame_features_length.get_info(info_name='op_quantile_0.95'))


class TFUsElecBertConverter(TFUsElecConverter):

    def training_preparation(self, examples, label_list):
        max_speech_length = NumericInfo(name='max_speech_length')
        max_frame_length = NumericInfo(name='max_frame_length')
        max_frame_features_length = NumericInfo(name='max_frame_features_length')

        for example in tqdm(examples):
            features = self.feature_class.convert_example(example=example,
                                                          label_list=label_list,
                                                          converter_info=self.get_info())

            speech_ids, speech_attention_mask, speech_mfcc, speech_audio, label_id = features

            speech_len = len(speech_ids) if speech_ids is not None else None

            frame_features_length = None
            frame_length = None
            if speech_mfcc is not None:
                frame_length = len(speech_mfcc)
                frame_features_length = speech_mfcc.shape[-1]

            if speech_audio is not None:
                frame_length = None
                frame_features_length = len(speech_audio)

            if speech_len is not None:
                max_speech_length.add(speech_len)

            if frame_length is not None:
                max_frame_length.add(frame_length)

            if frame_features_length is not None:
                max_frame_features_length.add(frame_features_length)

        max_speech_length.summary()
        max_speech_length.show()
        max_frame_length.summary()
        max_frame_length.show()
        max_frame_features_length.summary()
        max_frame_features_length.show()

        self.label_list = label_list
        self.max_speech_length = min(int(max_speech_length.get_info(info_name='op_quantile_0.95')), self.max_tokens_limit)
        self.max_frame_length = min(int(max_frame_length.get_info(info_name='op_quantile_0.95')), self.max_frame_limit)
        self.max_frame_features_length = int(max_frame_features_length.get_info(info_name='op_quantile_0.95'))


def register_us_elec_converter_components():
    ProjectRegistry.register_component(class_type=TFUsElecConverter,
                                       flag=ComponentFlag.CONVERTER,
                                       framework='tf',
                                       namespace='us_elec')

    ProjectRegistry.register_component(class_type=TFUsElecBertConverter,
                                       flag=ComponentFlag.CONVERTER,
                                       framework='tf',
                                       tags=['bert'],
                                       namespace='us_elec')


def register_converter_components():
    register_arg_aaai_converter_components()
    register_m_arg_converter_components()
    register_us_elec_converter_components()
