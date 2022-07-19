"""

Trains SVM baseline for MM-USElecDeb60to16 corpus

"""

import os

import librosa
import numpy as np
import pandas as pd
import resampy
from scipy import stats
from skimage.measure import block_reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import Wav2Vec2Processor, TFWav2Vec2Model

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_generic.utility.pickle_utils import load_pickle, save_pickle
from deasy_learning_generic.utility.python_utils import get_gridsearch_parameters
from sklearn.decomposition import TruncatedSVD
from deasy_learning_generic.commands import setup_registry
import tensorflow as tf
from deasy_learning_generic.utility.json_utils import save_json
from deasy_learning_generic.utility.log_utils import Logger

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def average_pooling(data, pooling_size):
    return block_reduce(data, (1, pooling_size), np.mean)


def load_audio_features(features, pooling_sizes=[], remove_energy=False, normalize=True):
    X = []
    for feature in features:
        if remove_energy:
            feature = feature[1:, :]

        if pooling_sizes is not None and len(pooling_sizes):
            for pooling_size in pooling_sizes:
                feature = average_pooling(feature, pooling_size)

        if normalize:
            feature = stats.zscore(feature, axis=1)

        svm_features = []
        for row in feature:
            svm_features.append(np.mean(row))
            svm_features.append(np.min(row))
            svm_features.append(np.max(row))
            svm_features.append(np.std(row))

        svm_features = np.array(svm_features)
        X.append(svm_features)

    # [# audio_files, # mfcc, values]
    return X


if __name__ == '__main__':

    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    setup_registry(directory=project_dir, module_names=['components', 'configurations'])

    # Helper
    framework_helper_config = ProjectRegistry.retrieve_configurations(flag=ComponentFlag.FRAMEWORK_HELPER,
                                                                      framework='tf',
                                                                      namespace='default',
                                                                      tags=['second_gpu'])
    helper = framework_helper_config.retrieve_component()
    helper.setup()

    seeds = [15371]

    task_type = 'acd'
    is_calibrating = False
    use_text_features = False
    use_audio_features = True
    use_audio_data = False
    audio_model_name = 'facebook/wav2vec2-base-960h'
    save_results = True
    test_name = 'svm_acd_audio_only'

    save_base_path = os.path.join(ProjectRegistry['task_dir'],
                                  'us_elec',
                                  test_name)

    if save_results and not os.path.isdir(save_base_path):
        os.makedirs(save_base_path)

    # Step 1: Settings

    # Loading data
    base_path = os.path.join(ProjectRegistry['local_database'], 'mm-uselecdeb60to16')
    df = pd.read_csv(os.path.join(base_path, 'final_dataset.csv'))

    if task_type == 'acd':
        df = df[df['Component'].isin(['Premise', 'Claim'])]
    else:
        df.loc[df['Component'].isin(['Premise', 'Claim']), 'Component'] = 'Arg'

    # Calibration
    # tuned_parameters = {
    #     'kernel': ['rbf', 'linear'],
    #     'gamma': [1, 5e-1, 1e-1, 5e-2, 1e-2],
    #     'C': [0.01, 0.1, 1, 10, 100]
    # }

    # ASD

    # Text-only
    # tuned_parameters = {
    #     'kernel': ['rbf'],
    #     'gamma': [0.5],
    #     'C': [0.1]
    # }

    # Audio-only
    # tuned_parameters = {
    #     'kernel': ['linear'],
    #     'gamma': [1],
    #     'C': [0.01]
    # }

    # Audio-only Wav2vec
    # tuned_parameters = {
    #     'kernel': ['rbf'],
    #     'gamma': [0.5],
    #     'C': [1]
    # }

    # Text-audio
    # tuned_parameters = {
    #     'kernel': ['linear'],
    #     'gamma': [1],
    #     'C': [0.01]
    # }

    # Text-audio Wav2vec
    # tuned_parameters = {
    #     'kernel': ['rbf'],
    #     'gamma': [0.1],
    #     'C': [1]
    # }

    # ACD

    # Text-only
    # tuned_parameters = {
    #     'kernel': ['rbf'],
    #     'gamma': [1],
    #     'C': [1]
    # }

    # Audio-only
    # tuned_parameters = {
    #     'kernel': ['linear'],
    #     'gamma': [1],
    #     'C': [0.1]
    # }

    # Audio-only Wav2vec
    # tuned_parameters = {
    #     'kernel': ['rbf'],
    #     'gamma': [0.01],
    #     'C': [100]
    # }

    # Text audio
    # tuned_parameters = {
    #     'kernel': ['linear'],
    #     'gamma': [1],
    #     'C': [100]
    # }

    # Text-audio Wav2vec
    tuned_parameters = {
        'kernel': ['linear'],
        'gamma': [1],
        'C': [0.1]
    }

    parameters_comb = get_gridsearch_parameters(tuned_parameters)

    text_key = 'Text'
    speech_key = 'audio_features'
    label_key = 'Component'
    label_map = {'O': 0, 'Arg': 1} if task_type == 'asd' else {'Premise': 0, 'Claim': 1}

    # Audio
    if use_audio_features:
        if not use_audio_data:
            audio_features_path = os.path.join(base_path, f'audio_features_{task_type}.pickle')
            audio_features = load_pickle(audio_features_path)
            df['audio_features'] = audio_features
        else:
            audio_sentences_path = os.path.join(base_path, 'audio_clips')
            audio_data_save_path = os.path.join(base_path, f'audio_data_{task_type}.pickle')
            if not os.path.isfile(audio_data_save_path):
                audio_processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path=audio_model_name)
                audio_model = TFWav2Vec2Model.from_pretrained(pretrained_model_name_or_path=audio_model_name)
                clip_ids = df['idClip'].values
                document_ids = df['Document'].values
                audio_data = []
                for clip_id, document_id in tqdm(zip(clip_ids, document_ids)):
                    speech_audio_file = os.path.join(audio_sentences_path, document_id, f'{clip_id}.wav')
                    speech_audio_emb_file = speech_audio_file.replace('.wav', '_emb.npy')
                    if not os.path.isfile(speech_audio_emb_file):
                        speech_audio_data, sample_rate = librosa.load(speech_audio_file, sr=None)
                        speech_audio_data = resampy.resample(speech_audio_data, sample_rate, 16000)
                        speech_audio_data = audio_processor(speech_audio_data, sampling_rate=16000).input_values[0]

                        speech_audio_embedding = audio_model(speech_audio_data[None, :]).last_hidden_state
                        speech_audio_embedding = np.mean(speech_audio_embedding.numpy(), axis=1).ravel()
                        np.save(speech_audio_emb_file, speech_audio_embedding)
                    else:
                        speech_audio_embedding = np.load(speech_audio_emb_file)

                    audio_data.append(speech_audio_embedding)

                # Flushing GPU
                del audio_processor
                del audio_model

                audio_data = np.array(audio_data)
                save_pickle(audio_data_save_path, audio_data)
            else:
                audio_data = load_pickle(audio_data_save_path)

            df['audio_features'] = audio_data.tolist()

    if not use_text_features and not use_audio_features:
        raise RuntimeError('Text or audio feature should be enable!')

    train_df = df[df['Set'] == 'TRAIN']
    val_df = df[df['Set'] == 'VALIDATION']
    test_df = df[df['Set'] == 'TEST']

    Logger.get_logger(__name__).info(f'Running SVM benchmark...')
    Logger.get_logger(__name__).info(f'Combinations: {len(parameters_comb)}')
    Logger.get_logger(__name__).info(f'Space: \n{tuned_parameters}')
    Logger.get_logger(__name__).info(f'task_type={task_type}')
    Logger.get_logger(__name__).info(f'is_calibration={is_calibrating}')
    Logger.get_logger(__name__).info(f'use_text_features={use_text_features}')
    Logger.get_logger(__name__).info(f'use_audio_features={use_audio_features}')
    Logger.get_logger(__name__).info(f'use_audio_data={use_audio_data}')

    best_comb = -1
    best_f1 = -1

    val_f1 = []
    test_f1 = []
    test_ablation_text_f1 = []
    test_ablation_audio_f1 = []
    for comb_idx, param_comb in tqdm(enumerate(parameters_comb)):
        seed_f1 = []

        for seed in seeds:
            np.random.seed(seed)

            train_texts = None
            test_texts = None
            val_texts = None
            train_audio = None
            test_audio = None
            val_audio = None

            x_train = None
            y_train = train_df[label_key].values
            y_train = np.array(list(map(lambda item: label_map[item], y_train)))
            x_val = None
            y_val = val_df[label_key].values
            y_val = np.array(list(map(lambda item: label_map[item], y_val)))
            x_test = None
            y_test = test_df[label_key].values
            y_test = np.array(list(map(lambda item: label_map[item], y_test)))

            if use_text_features:
                train_texts = train_df[text_key].values
                test_texts = test_df[text_key].values

                text_vectorizer = TfidfVectorizer()
                text_vectorizer.fit(train_texts)

                train_texts = text_vectorizer.transform(train_texts).toarray()
                test_texts = text_vectorizer.transform(test_texts).toarray()

                x_train = train_texts
                x_test = test_texts

                val_texts = val_df[text_key].values
                val_texts = text_vectorizer.transform(val_texts).toarray()
                x_val = val_texts

            if use_audio_features:
                train_audio = train_df['audio_features'].values
                if not use_audio_data:
                    train_audio = load_audio_features(features=train_audio,
                                                      pooling_sizes=[],
                                                      normalize=False,
                                                      remove_energy=False)
                train_audio = np.array([np.array(item) for item in train_audio])

                test_audio = test_df['audio_features'].values

                if not use_audio_data:
                    test_audio = load_audio_features(features=test_audio,
                                                     pooling_sizes=[],
                                                     normalize=False,
                                                     remove_energy=False)
                test_audio = np.array([np.array(item) for item in test_audio])

                val_audio = val_df['audio_features'].values

                if not use_audio_data:
                    val_audio = load_audio_features(features=val_audio,
                                                    pooling_sizes=[],
                                                    normalize=False,
                                                    remove_energy=False)
                val_audio = np.array([np.array(item) for item in val_audio])

                if train_texts is not None:
                    x_train = np.hstack((train_texts, train_audio))
                    x_test = np.hstack((test_texts, test_audio))
                    x_val = np.hstack((val_texts, val_audio))
                else:
                    x_train = train_audio
                    x_test = test_audio
                    x_val = val_audio

            svd = TruncatedSVD(n_components=100, random_state=42)
            x_train = svd.fit_transform(x_train)

            clf = SVC(class_weight="balanced", **param_comb)
            clf.fit(x_train, y_train)

            if is_calibrating:
                y_pred = clf.predict(svd.transform(x_val))
                y_true = y_val
            else:
                y_pred = clf.predict(svd.transform(x_test))
                y_true = y_test

            f1 = f1_score(y_true, y_pred, average='macro')
            seed_f1.append(f1)

            if not is_calibrating:
                fold_val_f1 = f1_score(y_true=y_val, y_pred=clf.predict(svd.transform(x_val)), average='macro')
                val_f1.append(fold_val_f1)

                fold_test_f1 = f1_score(y_true=y_test, y_pred=clf.predict(svd.transform(x_test)), average='macro')
                test_f1.append(fold_test_f1)

                if use_text_features and use_audio_features:
                    # Ablation study

                    # Text
                    fold_test_ablation_text_f1 = f1_score(y_true=y_test,
                                                          y_pred=clf.predict(svd.transform(np.hstack((np.zeros_like(
                                                              test_texts), test_audio)))),
                                                          average='macro')
                    test_ablation_text_f1.append(fold_test_ablation_text_f1)

                    # Audio
                    fold_test_ablation_audio_f1 = f1_score(y_true=y_test, y_pred=clf.predict(svd.transform(
                        np.hstack((test_texts, np.zeros_like(test_audio))))), average='macro')
                    test_ablation_audio_f1.append(fold_test_ablation_audio_f1)

        comb_f1 = np.mean(seed_f1)
        if best_f1 < comb_f1:
            best_f1 = comb_f1
            best_comb = comb_idx

    if is_calibrating:
        Logger.get_logger(__name__).info(f'Best comb: {parameters_comb[best_comb]}')
        Logger.get_logger(__name__).info(f'Best f1: {best_f1}')

        if save_results:
            save_json(os.path.join(save_base_path, 'calibration.json'), {
                'Best comb': parameters_comb[best_comb],
                'Best F1': best_f1,
                'Space': tuned_parameters
            })

    else:
        Logger.get_logger(__name__).info(f'Validation F1: {np.mean(val_f1)}')
        Logger.get_logger(__name__).info(f'Test F1: {np.mean(test_f1)}')

        to_save = {
            'is_calibrating': is_calibrating,
            'task_type': task_type,
            'use_text_features': use_text_features,
            'use_audio_features': use_audio_features,
            'use_audio_data': use_audio_data,
            'validation F1': np.mean(val_f1),
            'all validation F1': val_f1,
            'test F1': np.mean(test_f1),
            'all test F1': test_f1,
            'tuned_parameters': tuned_parameters
        }

        if use_audio_features and use_text_features:
            Logger.get_logger(__name__).info(f'Test Ablation-Text F1: {np.mean(test_ablation_text_f1)}')
            Logger.get_logger(__name__).info(f'Test Ablation-Audio F1: {np.mean(test_ablation_audio_f1)}')

            to_save['ablation text F1'] = np.mean(test_ablation_text_f1)
            to_save['ablation audio F1'] = np.mean(test_ablation_audio_f1)

        save_json(os.path.join(save_base_path, 'benchmark_info.json'), to_save)
