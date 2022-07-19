import os
import tarfile
from urllib import request

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from deasy_learning_generic.data_loader import DataLoader, DataSplit
from deasy_learning_generic.implementations.data_loader import DataFrameHandle
from deasy_learning_generic.implementations.labels import ClassificationLabel
from deasy_learning_generic.labels import LabelList
from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.pickle_utils import save_pickle, load_pickle


class ArgAAAILoader(DataLoader):

    def __init__(self, mode='all', mfccs=25, use_audio_features=True, **kwargs):
        super(ArgAAAILoader, self).__init__(**kwargs)
        self.download_path = os.path.join(ProjectRegistry.LOCAL_DATASETS_DIR, 'aaai2016_arg')
        self.has_loaded = os.path.isdir(self.download_path)
        self.dataset_name = 'aaai2016_arg'
        self.audio_sentences_path = os.path.join(self.download_path, 'audio')
        self.audio_features_path = os.path.join(self.download_path, 'audio_features.pickle')

        if mode == 'all':
            self.df_path = os.path.join(self.download_path, 'all_speakers.csv')
        else:
            self.df_path = os.path.join(self.download_path, '{}.csv'.format(mode.lower()))

        self.audio_path = os.path.join(self.download_path, 'mfcc')
        self.mode = mode
        self.mfccs = mfccs
        self.use_audio_features = use_audio_features

        # Currently disabled
        self.url = None

    def _download_dataset(self):
        os.makedirs(self.download_path)
        data_path = os.path.join(self.download_path, 'aaai2016_arg.tar.gz')

        # Download
        if not os.path.exists(data_path):
            request.urlretrieve(self.url, data_path)

        # Extract
        with tarfile.open(data_path) as loaded_tar:
            loaded_tar.extractall(self.download_path)

        # Clean
        os.remove(data_path)

    def load(self):
        if not self.has_loaded:
            Logger.get_logger(__name__).info('Loading AAAI2016_Arg dataset...')
            self.has_loaded = os.path.isdir(self.download_path)

        df = pd.read_csv(self.df_path)
        return df

    def _load_audio_features(self, df):
        audio_features = []
        # Loop through the whole dataframe that extracts the audio features of
        # the first and second sentences of the pair
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                speaker = row['speaker_id']
                x, sr = librosa.load(os.path.join(self.audio_sentences_path, speaker, f'{row["mfccs_id"]}.wav'))
                mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=self.mfccs)[2:]
                spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
                chroma_ft = librosa.feature.chroma_stft(y=x, sr=sr)
                features = np.concatenate(
                    (spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),
                    axis=0)
                audio_features.append(features)
            except:
                # this is for the case when the audio sentences have 0 duration
                # (there are some because of the alignment software interacting with certain complicated situations)
                df = df.drop(index=index)
                Logger.get_logger(__name__).info("Removed from dataset due to faulty audio feature extraction.")
                continue

        return audio_features

    def build_data_splits(self):
        df = self.load()

        if self.use_audio_features:
            # Load audio features
            # List of samples, each being a np.ndarray of shape (mfccs, #sample_frames)
            if not os.path.isfile(self.audio_features_path):
                Logger.get_logger(__name__).info('Building audio features with librosa...')
                audio_features = self._load_audio_features(df=df)
                save_pickle(self.audio_features_path, audio_features)
            else:
                audio_features = load_pickle(self.audio_features_path)
            df['audio_features'] = audio_features
        else:
            audio_filename_paths = [os.path.join(self.audio_sentences_path, speaker_id, f'{mfccs_id}.wav')
                                    for speaker_id, mfccs_id in zip(df['speaker_id'].values, df['mfccs_id'].values)]
            df['audio_filename_paths'] = audio_filename_paths

        # Labels
        labels = LabelList([
            ClassificationLabel(name='sentence',
                                values=[0, 1])
        ])

        # Data splits
        train_data = DataFrameHandle(data=df, split=DataSplit.TRAIN, data_name='ArgAAAI2016',
                                     data_keys={
                                         'text': 'text',
                                         'audio': 'audio_features',
                                         'audio_file': 'audio_filename_paths',
                                         'sentence': 'label'
                                     },
                                     labels=labels)
        self.assign_train_data(data=train_data)


class MArgLoader(DataLoader):

    def __init__(self, annotation_confidence=0., mfccs=25, use_audio_features=True, **kwargs):
        super(MArgLoader, self).__init__(**kwargs)
        self.annotation_confidence = annotation_confidence
        self.mfccs = mfccs
        self.use_audio_features = use_audio_features

        self.download_path = os.path.join(ProjectRegistry.LOCAL_DATASETS_DIR, 'm-arg')
        self.has_loaded = os.path.isdir(self.download_path)
        self.dataset_name = 'm-arg'
        self.feature_df_path = os.path.join(self.download_path, 'preprocessed full dataset',
                                            'full_feature_extraction_dataset.csv')
        self.aggregated_df_path = os.path.join(self.download_path, 'aggregated_dataset.csv')
        self.audio_sentences_path = os.path.join(self.download_path, 'audio sentences')
        self.final_df_path = os.path.join(self.download_path, 'final_dataset_{:.2f}.csv')
        self.audio_features_path = os.path.join(self.download_path, 'audio_features{0}_{1}_{2:.2f}.pickle')

        # Currently disabled
        self.url = None

    def _download_dataset(self):
        os.makedirs(self.download_path)
        data_path = os.path.join(self.download_path, 'm-arg.tar.gz')

        # Download
        if not os.path.exists(data_path):
            request.urlretrieve(self.url, data_path)

        # Extract
        with tarfile.open(data_path) as loaded_tar:
            loaded_tar.extractall(self.download_path)

        # Clean
        os.remove(data_path)

    def _build_complete_dataset(self, feature_df, aggregated_df):
        df_final = pd.DataFrame(columns=['id', 'relation', 'confidence',
                                         'sentence_1', 'sentence_2',
                                         'sentence_1_audio', 'sentence_2_audio'])

        for index, row in aggregated_df.iterrows():
            # ids
            id1 = row["pair_id"]

            # labels
            relation1 = row["relation"]

            # label confidence
            conf1 = row["relation:confidence"]

            # sentences
            s1t = row["sentence_1"]
            s2t = row["sentence_2"]

            # correponding audio sentences based on the text
            s1a = feature_df['audio_file'].loc[feature_df['text'] == s1t].values[0]
            s2a = feature_df['audio_file'].loc[feature_df['text'] == s2t].values[0]

            # If we want to filter by annotation confidence we can add here the following if statement
            if row["relation:confidence"] > self.annotation_confidence:
                df_final = df_final.append(
                    {'id': id1, 'relation': relation1, 'confidence': conf1, 'sentence_1': s1t, 'sentence_2': s2t,
                     'sentence_1_audio': s1a, 'sentence_2_audio': s2a}, ignore_index=True)

        return df_final

    def load(self):
        if not self.has_loaded:
            Logger.get_logger(__name__).info('Loading M-Arg dataset...')
            self.has_loaded = os.path.isdir(self.download_path)

        if not os.path.isfile(self.final_df_path.format(self.annotation_confidence)):
            feature_df = pd.read_csv(self.feature_df_path)
            aggregated_df = pd.read_csv(self.aggregated_df_path)
            train_df = self._build_complete_dataset(feature_df=feature_df,
                                                    aggregated_df=aggregated_df)

            # Add index for cv routine
            train_df['index'] = np.arange(train_df.shape[0])

            train_df.to_csv(self.final_df_path.format(self.annotation_confidence), index=False)
        else:
            train_df = pd.read_csv(self.final_df_path.format(self.annotation_confidence))

        return train_df

    def _load_audio_features(self, df):
        audio_features1 = []
        audio_features2 = []

        # Loop through the whole dataframe that extracts the audio features of
        # the first and second sentences of the pair
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                # first sentence
                x, sr = librosa.load(os.path.join(self.audio_sentences_path, row['sentence_1_audio'] + '.wav'))
                mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=self.mfccs)[2:]
                spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
                chroma_ft = librosa.feature.chroma_stft(y=x, sr=sr)
                features = np.concatenate(
                    (spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),
                    axis=0)
                audio_features1.append(features)
            except:
                # this is for the case when the audio sentences have 0 duration
                # (there are some because of the alignment software interracting with certain complicated situations)
                df = df.drop(index=index)
                Logger.get_logger(__name__).info("Pair removed from dataset due to faulty audio feature extraction.")
                continue

                # second sentence
            try:
                x, sr = librosa.load(os.path.join(self.audio_sentences_path, row['sentence_2_audio'] + '.wav'))
                mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=self.mfccs)[2:]
                spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
                chroma_ft = librosa.feature.chroma_stft(y=x, sr=sr)
                features = np.concatenate(
                    (spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),
                    axis=0)
                audio_features2.append(features)
            except:
                # this is for the case when the audio sentences have 0 duration
                # (there are some because of the alignment software interracting with certain complicated situations)
                df = df.drop(index=index)
                Logger.get_logger(__name__).info("Pair removed from dataset due to faulty audio feature extraction.")
                audio_features1.pop()

        return audio_features1, audio_features2

    def build_data_splits(self):
        train_df = self.load()

        # Load audio features
        # List of samples, each being a np.ndarray of shape (mfccs, #sample_frames)
        if self.use_audio_features:
            if not os.path.isfile(self.audio_features_path.format('1', self.mfccs, self.annotation_confidence)):
                Logger.get_logger(__name__).info('Building audio features with librosa...')
                audio_features1, audio_features2 = self._load_audio_features(df=train_df)
                save_pickle(self.audio_features_path.format('1', self.mfccs, self.annotation_confidence),
                            audio_features1)
                save_pickle(self.audio_features_path.format('2', self.mfccs, self.annotation_confidence),
                            audio_features2)
            else:
                audio_features1 = load_pickle(
                    self.audio_features_path.format('1', self.mfccs, self.annotation_confidence))
                audio_features2 = load_pickle(
                    self.audio_features_path.format('2', self.mfccs, self.annotation_confidence))

            train_df['audio_features1'] = audio_features1
            train_df['audio_features2'] = audio_features2
        else:
            audio_a_filename_paths = [os.path.join(self.audio_sentences_path, sentence + '.wav') for sentence in
                                      train_df['sentence_1_audio'].values]
            audio_b_filename_paths = [os.path.join(self.audio_sentences_path, sentence + '.wav') for sentence in
                                      train_df['sentence_2_audio'].values]

            train_df['audio_a_filename_paths'] = audio_a_filename_paths
            train_df['audio_b_filename_paths'] = audio_b_filename_paths

        labels = LabelList([
            ClassificationLabel(name='relation',
                                values=['neither', 'support', 'attack'])
        ])

        train_data = DataFrameHandle(data=train_df, split=DataSplit.TRAIN, data_name='M-Arg',
                                     data_keys={
                                         'text_a': 'sentence_1',
                                         'text_b': 'sentence_2',
                                         'audio_a': 'audio_features1',
                                         'audio_b': 'audio_features2',
                                         'audio_a_filename_paths': 'audio_a_filename_paths',
                                         'audio_b_filename_paths': 'audio_b_filename_paths',
                                         'relation': 'relation'
                                     },
                                     labels=labels)
        self.assign_train_data(data=train_data)


class UsElecLoader(DataLoader):

    def __init__(self, task_type='asd', data_mode='text_only', mfccs=25, use_audio_features=True, **kwargs):
        super(UsElecLoader, self).__init__(**kwargs)
        assert task_type in ['asd', 'acd']
        self.task_type = task_type
        self.mfccs = mfccs
        self.use_audio_features = use_audio_features
        self.data_mode = data_mode

        self.download_path = os.path.join(ProjectRegistry.LOCAL_DATASETS_DIR, 'mm-uselecdeb60to16')
        self.has_loaded = os.path.isdir(self.download_path)
        self.dataset_name = 'mm-uselecdeb60to16'
        self.audio_sentences_path = os.path.join(self.download_path, 'audio_clips')
        self.df_path = os.path.join(self.download_path, 'final_dataset.csv')
        self.audio_features_path = os.path.join(self.download_path, 'audio_features_{}.pickle'.format(task_type))

        # Currently disabled
        self.url = None

    def _download_dataset(self):
        os.makedirs(self.download_path)
        data_path = os.path.join(self.download_path, f'{self.dataset_name}.tar.gz')

        # Download
        if not os.path.exists(data_path):
            request.urlretrieve(self.url, data_path)

        # Extract
        with tarfile.open(data_path) as loaded_tar:
            loaded_tar.extractall(self.download_path)

        # Clean
        os.remove(data_path)

    def load(self):
        if not self.has_loaded:
            Logger.get_logger(__name__).info('Loading MM-USElecDeb60to16 dataset...')
            self.has_loaded = os.path.isdir(self.download_path)

        df = pd.read_csv(self.df_path)
        return df

    def _load_audio_features(self, df):
        audio_features = []
        # Loop through the whole dataframe that extracts the audio features of
        # the first and second sentences of the pair
        failed = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                document_id = row['Document']
                feature_save_path = os.path.join(self.audio_sentences_path, document_id, f'{row["idClip"]}_mfccs.npy')

                if not os.path.isfile(feature_save_path):
                    x, sr = librosa.load(os.path.join(self.audio_sentences_path, document_id, f'{row["idClip"]}.wav'))
                    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=self.mfccs)[2:]
                    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
                    spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
                    chroma_ft = librosa.feature.chroma_stft(y=x, sr=sr)
                    features = np.concatenate(
                        (spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),
                        axis=0)
                    np.save(feature_save_path, features)
                else:
                    features = np.load(feature_save_path)
                audio_features.append(features)
            except Exception as e:
                # this is for the case when the audio sentences have 0 duration
                # (there are some because of the alignment software interacting with certain complicated situations)
                df = df.drop(index=index)
                Logger.get_logger(__name__).info(f"Removed sample idx={index} from dataset due to faulty audio feature extraction. "
                                                 f"Reasons: {e}")
                failed.append(index)

        if len(failed):
            Logger.get_logger(__name__).info(f'Total failed samples: {len(failed)}')
            np.save(os.path.join(self.audio_sentences_path, 'failed_list.npy'), failed)
        return audio_features

    def build_data_splits(self):
        df = self.load()

        if self.task_type == 'acd':
            df = df[df['Component'].isin(['Premise', 'Claim'])]
        else:
            df.loc[df['Component'].isin(['Premise', 'Claim']), 'Component'] = 'Arg'

        if self.data_mode != 'text_only':
            if self.use_audio_features:
                # Load audio features
                # List of samples, each being a np.ndarray of shape (mfccs, #sample_frames)
                if not os.path.isfile(self.audio_features_path):
                    Logger.get_logger(__name__).info('Building audio features with librosa...')
                    audio_features = self._load_audio_features(df=df)
                    save_pickle(self.audio_features_path, audio_features)
                else:
                    audio_features = load_pickle(self.audio_features_path)
                df['audio_features'] = audio_features
            else:
                audio_filename_paths = [os.path.join(self.audio_sentences_path, document_id, f'{clip_id}.wav')
                                        for document_id, clip_id in zip(df['Document'].values, df['idClip'].values)]
                df['audio_filename_paths'] = audio_filename_paths

        # Labels
        if self.task_type == 'asd':
            labels = LabelList([
                ClassificationLabel(name='sentence',
                                    values=['O', 'Arg'])
            ])
        else:
            labels = LabelList([
                ClassificationLabel(name='component',
                                    values=['Premise', 'Claim'])
            ])

        train_df = df[df['Set'] == 'TRAIN']
        val_df = df[df['Set'] == 'VALIDATION']
        test_df = df[df['Set'] == 'TEST']

        # Data splits
        train_data = DataFrameHandle(data=train_df, split=DataSplit.TRAIN, data_name=f'MM-USElecDeb60to16_{self.task_type}',
                                     data_keys={
                                         'text': 'Text',
                                         'audio': 'audio_features',
                                         'audio_file': 'audio_filename_paths',
                                         'sentence': 'Component',
                                         'component': 'Component'
                                     },
                                     labels=labels)
        self.assign_train_data(data=train_data)

        val_data = DataFrameHandle(data=val_df, split=DataSplit.VAL, data_name=f'MM-USElecDeb60to16_{self.task_type}',
                                   data_keys={
                                       'text': 'Text',
                                       'audio': 'audio_features',
                                       'audio_file': 'audio_filename_paths',
                                       'sentence': 'Component',
                                       'component': 'Component'
                                   },
                                   labels=labels)
        self.assign_val_data(data=val_data)

        test_data = DataFrameHandle(data=test_df, split=DataSplit.TEST, data_name=f'MM-USElecDeb60to16_{self.task_type}',
                                    data_keys={
                                        'text': 'Text',
                                        'audio': 'audio_features',
                                        'audio_file': 'audio_filename_paths',
                                        'sentence': 'Component',
                                        'component': 'Component'
                                    },
                                    labels=labels)
        self.assign_test_data(data=test_data)


def register_data_loader_components():
    ProjectRegistry.register_component(class_type=ArgAAAILoader,
                                       flag=ComponentFlag.DATA_LOADER,
                                       framework='generic',
                                       namespace='arg_aaai')

    ProjectRegistry.register_component(class_type=MArgLoader,
                                       flag=ComponentFlag.DATA_LOADER,
                                       framework='generic',
                                       namespace='m-arg')

    ProjectRegistry.register_component(class_type=UsElecLoader,
                                       flag=ComponentFlag.DATA_LOADER,
                                       framework='generic',
                                       namespace='us_elec')
