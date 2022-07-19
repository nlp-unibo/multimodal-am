from tqdm import tqdm

from components.examples import ArgAAAIExample, MArgExample, UsElecExample
from deasy_learning_generic.examples import ExampleList
from deasy_learning_generic.nlp.utility import preprocessing_utils
from deasy_learning_generic.processor import DataProcessor
from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from utility import speech_utils
import librosa
import resampy
from transformers import Wav2Vec2Processor, TFWav2Vec2Model
import numpy as np
import os


class ArgAAAIProcessor(DataProcessor):

    def __init__(self, data_mode='text_only', pooling_sizes=None,
                 normalize_mfccs=True, use_audio_features=True, audio_model_sampling_rate=None,
                 audio_model_name=None, **kwargs):
        super(ArgAAAIProcessor, self).__init__(**kwargs)
        self.data_mode = data_mode
        self.pooling_sizes = pooling_sizes
        self.normalize_mfccs = normalize_mfccs
        self.use_audio_features = use_audio_features
        self.audio_model_sampling_rate = audio_model_sampling_rate
        self.audio_model_name = audio_model_name

        if not use_audio_features:
            assert audio_model_name is not None

    def _get_examples(self, data, audio_processor, audio_model):
        examples = ExampleList()

        data_keys = data.get_data_keys()
        assert 'text' in data_keys
        assert 'audio' in data_keys
        assert 'sentence' in data_keys

        total_items = len(data)
        for item_idx in tqdm(range(total_items)):
            item = data[item_idx]

            # Speech text
            if self.data_mode != 'audio_only':
                speech_text = data.get_item_value(item_idx=item_idx,
                                                  value_key=data_keys['text'])
                speech_text = preprocessing_utils.filter_line(speech_text,
                                                              function_names=self.filter_names,
                                                              disable_filtering=self.disable_filtering)
            else:
                speech_text = None

            # Speech audio
            speech_mfcc = None
            speech_audio_data = None
            if self.data_mode in ['audio_only', 'text_audio']:
                if self.use_audio_features:
                    speech_mfcc = data.get_item_value(item_idx=item_idx,
                                                      value_key=data_keys['audio'])

                    # [frames, #mfccs]
                    speech_mfcc = speech_mfcc.transpose()
                    speech_mfcc = speech_utils.parse_audio_features(data=speech_mfcc,
                                                                    pooling_sizes=self.pooling_sizes,
                                                                    remove_energy=False)

                    if self.normalize_mfccs:
                        speech_mfcc = speech_utils.normalize_speaker_audio(data=speech_mfcc)
                else:
                    speech_audio_file = data.get_item_value(item_idx=item_idx,
                                                            value_key=data_keys['audio_file'])
                    speech_audio_emb_file = speech_audio_file.replace('.wav', '_emb.npy')
                    if not os.path.isfile(speech_audio_emb_file):
                        speech_audio_data, sample_rate = librosa.load(speech_audio_file, sr=None)
                        speech_audio_data = resampy.resample(speech_audio_data, sample_rate, self.audio_model_sampling_rate)
                        speech_audio_data = audio_processor(speech_audio_data,
                                                            sampling_rate=self.audio_model_sampling_rate).input_values[0]
                        speech_audio_data = audio_model(speech_audio_data[None, :]).last_hidden_state
                        speech_audio_data = np.mean(speech_audio_data.numpy().squeeze(axis=0), axis=0)
                        np.save(speech_audio_emb_file, speech_audio_data)
                    else:
                        speech_audio_data = np.load(speech_audio_emb_file)

            # Label
            if data.has_labels():
                label = self._retrieve_default_label(labels=data.get_labels(), item=item, data_keys=data_keys)
            else:
                label = None

            assert speech_text is not None or speech_mfcc is not None or speech_audio_data is not None
            example = ArgAAAIExample(speech_text=speech_text, speech_mfccs=speech_mfcc,
                                     speech_audio_data=speech_audio_data, label=label)
            examples.append(example)

        return examples

    def _transform_data(self, data, model_path, suffix, save_prefix=None, component_info=None, filepath=None):
        audio_model = None
        audio_processor = None
        if not self.use_audio_features:
            audio_processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path=self.audio_model_name)
            audio_model = TFWav2Vec2Model.from_pretrained(pretrained_model_name_or_path=self.audio_model_name)

        examples = self._get_examples(data=data, audio_processor=audio_processor, audio_model=audio_model)

        del audio_processor
        del audio_model

        return examples


class MArgProcessor(DataProcessor):

    def __init__(self, data_mode='text_only', pooling_sizes=None,
                 normalize_mfccs=True, use_audio_features=True, audio_model_sampling_rate=None,
                 audio_model_name=None, **kwargs):
        super(MArgProcessor, self).__init__(**kwargs)
        self.data_mode = data_mode
        self.pooling_sizes = pooling_sizes
        self.normalize_mfccs = normalize_mfccs
        self.use_audio_features = use_audio_features
        self.audio_model_sampling_rate = audio_model_sampling_rate
        self.audio_model_name = audio_model_name

        if not use_audio_features:
            assert audio_model_name is not None

    def _get_examples(self, data, audio_processor, audio_model):
        examples = ExampleList()

        data_keys = data.get_data_keys()
        assert 'text_a' in data_keys
        assert 'text_b' in data_keys
        assert 'audio_a' in data_keys
        assert 'audio_b' in data_keys
        assert 'relation' in data_keys

        total_items = len(data)
        for item_idx in tqdm(range(total_items)):
            item = data[item_idx]

            # Speech text
            if self.data_mode != 'audio_only':
                text_a = data.get_item_value(item_idx=item_idx, value_key=data_keys['text_a'])
                text_a = preprocessing_utils.filter_line(text_a,
                                                         function_names=self.filter_names,
                                                         disable_filtering=self.disable_filtering)

                text_b = data.get_item_value(item_idx=item_idx, value_key=data_keys['text_b'])
                text_b = preprocessing_utils.filter_line(text_b,
                                                         function_names=self.filter_names,
                                                         disable_filtering=self.disable_filtering)
            else:
                text_a = None
                text_b = None

            # Speech audio
            audio_a = None
            audio_a_data = None
            audio_b = None
            audio_b_data = None
            if self.data_mode in ['audio_only', 'text_audio']:
                if self.use_audio_features:
                    audio_a = data.get_item_value(item_idx=item_idx,
                                                  value_key=data_keys['audio_a'])

                    # [frames, #mfccs]
                    audio_a = audio_a.transpose()
                    audio_a = speech_utils.parse_audio_features(data=audio_a,
                                                                pooling_sizes=self.pooling_sizes,
                                                                remove_energy=False)

                    if self.normalize_mfccs:
                        audio_a = speech_utils.normalize_speaker_audio(data=audio_a)

                    audio_b = data.get_item_value(item_idx=item_idx,
                                                  value_key=data_keys['audio_b'])

                    # [frames, #mfccs]
                    audio_b = audio_b.transpose()
                    audio_b = speech_utils.parse_audio_features(data=audio_b,
                                                                pooling_sizes=self.pooling_sizes,
                                                                remove_energy=False)

                    if self.normalize_mfccs:
                        audio_b = speech_utils.normalize_speaker_audio(data=audio_b)
                else:
                    audio_a_audio_file = data.get_item_value(item_idx=item_idx,
                                                             value_key=data_keys['audio_a_filename_paths'])
                    audio_a_audio_emb_file = audio_a_audio_file.replace('.wav', '_emb.npy')
                    if not os.path.isfile(audio_a_audio_emb_file):
                        audio_a_data, sample_rate = librosa.load(audio_a_audio_file, sr=None)
                        audio_a_data = resampy.resample(audio_a_data, sample_rate, self.audio_model_sampling_rate)
                        audio_a_data = audio_processor(audio_a_data, sampling_rate=self.audio_model_sampling_rate).input_values[0]
                        audio_a_data = audio_model(audio_a_data[None, :]).last_hidden_state
                        audio_a_data = np.mean(audio_a_data.numpy().squeeze(axis=0), axis=0)
                        np.save(audio_a_audio_emb_file, audio_a_data)
                    else:
                        audio_a_data = np.load(audio_a_audio_emb_file)

                    audio_b_audio_file = data.get_item_value(item_idx=item_idx,
                                                             value_key=data_keys['audio_b_filename_paths'])
                    audio_b_audio_emb_file = audio_a_audio_file.replace('.wav', '_emb.npy')
                    if not os.path.isfile(audio_b_audio_emb_file):
                        audio_b_data, sample_rate = librosa.load(audio_b_audio_file, sr=None)
                        audio_b_data = resampy.resample(audio_b_data, sample_rate, self.audio_model_sampling_rate)
                        audio_b_data = audio_processor(audio_b_data, sampling_rate=self.audio_model_sampling_rate).input_values[0]
                        audio_b_data = audio_model(audio_b_data[None, :]).last_hidden_state
                        audio_b_data = np.mean(audio_b_data.numpy().squeeze(axis=0), axis=0)
                        np.save(audio_b_audio_emb_file, audio_b_data)
                    else:
                        audio_b_data = np.load(audio_b_audio_emb_file)

            # Label
            if data.has_labels():
                label = self._retrieve_default_label(labels=data.get_labels(), item=item, data_keys=data_keys)
            else:
                label = None

            assert text_a is not None or audio_a is not None or audio_a_data is not None
            assert text_b is not None or audio_b is not None or audio_b_data is not None
            example = MArgExample(text_a=text_a, text_b=text_b,
                                  audio_a=audio_a, audio_b=audio_b,
                                  audio_a_data=audio_a_data, audio_b_data=audio_b_data,
                                  label=label)
            examples.append(example)

        return examples

    def _transform_data(self, data, model_path, suffix, save_prefix=None, component_info=None, filepath=None):
        audio_model = None
        audio_processor = None
        if not self.use_audio_features:
            audio_processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path=self.audio_model_name)
            audio_model = TFWav2Vec2Model.from_pretrained(pretrained_model_name_or_path=self.audio_model_name)

        examples = self._get_examples(data=data, audio_processor=audio_processor, audio_model=audio_model)
        del audio_processor
        del audio_model

        return examples


class USElecProcessor(DataProcessor):

    def __init__(self, data_mode='text_only', pooling_sizes=None,
                 normalize_mfccs=True, use_audio_features=True, audio_model_sampling_rate=None,
                 audio_model_name=None, **kwargs):
        super(USElecProcessor, self).__init__(**kwargs)
        self.data_mode = data_mode
        self.pooling_sizes = pooling_sizes
        self.normalize_mfccs = normalize_mfccs
        self.use_audio_features = use_audio_features
        self.audio_model_sampling_rate = audio_model_sampling_rate
        self.audio_model_name = audio_model_name

        if not use_audio_features:
            assert audio_model_name is not None

    def _get_examples(self, data, audio_processor, audio_model):
        examples = ExampleList()

        data_keys = data.get_data_keys()
        assert 'text' in data_keys
        assert 'audio' in data_keys
        assert 'sentence' in data_keys
        assert 'component' in data_keys

        total_items = len(data)
        for item_idx in tqdm(range(total_items)):
            item = data[item_idx]

            # Speech text
            if self.data_mode != 'audio_only':
                speech_text = data.get_item_value(item_idx=item_idx,
                                                  value_key=data_keys['text'])
                speech_text = preprocessing_utils.filter_line(speech_text,
                                                              function_names=self.filter_names,
                                                              disable_filtering=self.disable_filtering)
            else:
                speech_text = None

            # Speech audio
            speech_mfcc = None
            speech_audio_data = None
            if self.data_mode in ['audio_only', 'text_audio']:
                if self.use_audio_features:
                    speech_mfcc = data.get_item_value(item_idx=item_idx,
                                                      value_key=data_keys['audio'])

                    # [frames, #mfccs]
                    speech_mfcc = speech_mfcc.transpose()
                    speech_mfcc = speech_utils.parse_audio_features(data=speech_mfcc,
                                                                    pooling_sizes=self.pooling_sizes,
                                                                    remove_energy=False)

                    if self.normalize_mfccs:
                        speech_mfcc = speech_utils.normalize_speaker_audio(data=speech_mfcc)
                else:
                    speech_audio_file = data.get_item_value(item_idx=item_idx,
                                                            value_key=data_keys['audio_file'])
                    speech_audio_emb_file = speech_audio_file.replace('.wav', '_emb.npy')
                    if not os.path.isfile(speech_audio_emb_file):
                        speech_audio_data, sample_rate = librosa.load(speech_audio_file, sr=None)
                        speech_audio_data = resampy.resample(speech_audio_data, sample_rate, self.audio_model_sampling_rate)
                        speech_audio_data = audio_processor(speech_audio_data,
                                                            sampling_rate=self.audio_model_sampling_rate).input_values[0]
                        speech_audio_data = audio_model(speech_audio_data[None, :]).last_hidden_state
                        speech_audio_data = np.mean(speech_audio_data.numpy().squeeze(axis=0), axis=0)
                        np.save(speech_audio_emb_file, speech_audio_data)
                    else:
                        speech_audio_data = np.load(speech_audio_emb_file)

            # Label
            if data.has_labels():
                label = self._retrieve_default_label(labels=data.get_labels(), item=item, data_keys=data_keys)
            else:
                label = None

            assert speech_text is not None or speech_mfcc is not None or speech_audio_data is not None
            example = UsElecExample(speech_text=speech_text, speech_mfccs=speech_mfcc,
                                     speech_audio_data=speech_audio_data, label=label)
            examples.append(example)

        return examples

    def _transform_data(self, data, model_path, suffix, save_prefix=None, component_info=None, filepath=None):
        audio_model = None
        audio_processor = None
        if not self.use_audio_features:
            audio_processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path=self.audio_model_name)
            audio_model = TFWav2Vec2Model.from_pretrained(pretrained_model_name_or_path=self.audio_model_name)

        examples = self._get_examples(data=data, audio_processor=audio_processor, audio_model=audio_model)

        del audio_processor
        del audio_model

        return examples


def register_processor_components():
    ProjectRegistry.register_component(class_type=ArgAAAIProcessor,
                                       flag=ComponentFlag.PROCESSOR,
                                       framework='generic',
                                       namespace='arg_aaai')

    ProjectRegistry.register_component(class_type=MArgProcessor,
                                       flag=ComponentFlag.PROCESSOR,
                                       framework='generic',
                                       namespace='m-arg')

    ProjectRegistry.register_component(class_type=USElecProcessor,
                                       flag=ComponentFlag.PROCESSOR,
                                       framework='generic',
                                       namespace='us_elec')
