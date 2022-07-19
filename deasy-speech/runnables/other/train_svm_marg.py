"""

Trains SVM baseline for M-Arg corpus

"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from tqdm import tqdm
from scipy import stats
from skimage.measure import block_reduce
from transformers import Wav2Vec2Processor, TFWav2Vec2Model
import librosa
import resampy

from deasy_learning_generic.registry import ProjectRegistry
from deasy_learning_generic.utility.pickle_utils import load_pickle, save_pickle
from deasy_learning_generic.utility.python_utils import get_gridsearch_parameters
from deasy_learning_generic.utility.routine_utils import PrebuiltCV
from sklearn.decomposition import TruncatedSVD
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
    ProjectRegistry.set_project_dir(project_dir)
    seeds = [15371]

    is_calibrating = False
    confidence = 0.85
    use_text_features = True
    use_audio_features = True
    use_audio_data = False
    audio_model_name = 'facebook/wav2vec2-base-960h'
    save_results = True
    test_name = 'svm_085_text_audio'

    save_base_path = os.path.join(ProjectRegistry['task_dir'],
                                  'm-arg',
                                  test_name)

    if save_results and not os.path.isdir(save_base_path):
        os.makedirs(save_base_path)

    # Step 1: Settings

    # Loading data
    base_path = os.path.join(ProjectRegistry['local_database'], 'm-arg')
    df = pd.read_csv(os.path.join(base_path, 'final_dataset_{:.2f}.csv'.format(confidence)))

    # Calibration
    # tuned_parameters = {
    #     'kernel': ['rbf', 'linear'],
    #     'gamma': [1, 5e-1, 1e-1, 5e-2, 1e-2],
    #     'C': [0.01, 0.1, 1, 10, 100]
    # }

    # Text-only
    # tuned_parameters = {
    #     'kernel': ['linear'],
    #     'gamma': [1],
    #     'C': [100]
    # }

    # Audio-only
    # tuned_parameters = {
    #     'kernel': ['linear'],
    #     'gamma': [1],
    #     'C': [0.01]
    # }

    # Audio-only (wav2vec)
    # tuned_parameters = {
    #     'kernel': ['rbf'],
    #     'gamma': [0.1],
    #     'C': [1]
    # }

    # Text audio
    tuned_parameters = {
        'kernel': ['linear'],
        'gamma': [1],
        'C': [0.01]
    }

    # Text audio (wav2vec)
    # tuned_parameters = {
    #     'kernel': ['rbf'],
    #     'gamma': [0.5],
    #     'C': [1]
    # }

    parameters_comb = get_gridsearch_parameters(tuned_parameters)

    text_a_key = 'sentence_1'
    text_b_key = 'sentence_2'
    speech_a_key = 'audio_features1'
    speech_b_key = 'audio_features2'
    label_key = 'relation'

    # Audio
    if not use_audio_data:
        audio_features_path = os.path.join(ProjectRegistry['local_database'], 'm-arg')
        audio_features1 = load_pickle(
            os.path.join(audio_features_path, 'audio_features1_25_{:.2f}.pickle'.format(confidence)))
        audio_features2 = load_pickle(
            os.path.join(audio_features_path, 'audio_features2_25_{:.2f}.pickle'.format(confidence)))
        df['audio_features1'] = audio_features1
        df['audio_features2'] = audio_features2
    else:
        audio_sentences_path = os.path.join(ProjectRegistry['local_database'], 'm-arg', 'audio sentences')
        audio_data_a_save_path = os.path.join(ProjectRegistry['local_database'], 'm-arg',
                                              'audio_data_a_{:.2f}.pickle'.format(confidence))
        audio_data_b_save_path = os.path.join(ProjectRegistry['local_database'], 'm-arg',
                                              'audio_data_b_{:.2f}.pickle'.format(confidence))

        audio_processor = None
        audio_model = None

        # A
        if not os.path.isfile(audio_data_a_save_path):
            audio_processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path=audio_model_name)
            audio_model = TFWav2Vec2Model.from_pretrained(pretrained_model_name_or_path=audio_model_name)
            audio_files = df['sentence_1_audio'].values
            audio_data_a = []
            for audio_file in tqdm(audio_files):
                speech_audio_file = os.path.join(audio_sentences_path, audio_file + '.wav')
                speech_audio_emb_file = speech_audio_file.replace('.wav', '_emb.npy')
                if not os.path.isfile(speech_audio_emb_file):
                    speech_audio_data, sample_rate = librosa.load(speech_audio_file, sr=None)
                    speech_audio_data = resampy.resample(speech_audio_data, sample_rate, 16000)
                    speech_audio_data = audio_processor(speech_audio_data, sampling_rate=16000).input_values[0]

                    speech_audio_embedding = audio_model(speech_audio_data[None, :]).last_hidden_state
                    speech_audio_embedding = np.mean(speech_audio_embedding.numpy(), axis=1).squeeze()
                    np.save(speech_audio_emb_file, speech_audio_embedding)
                else:
                    speech_audio_embedding = np.load(speech_audio_emb_file)

                audio_data_a.append(speech_audio_embedding)

            audio_data_a = np.array(audio_data_a)
            save_pickle(audio_data_a_save_path, audio_data_a)
        else:
            audio_data_a = load_pickle(audio_data_a_save_path)

        df['audio_features1'] = audio_data_a.tolist()

        # B
        if not os.path.isfile(audio_data_b_save_path):
            audio_files = df['sentence_2_audio'].values
            audio_data_b = []
            for audio_file in tqdm(audio_files):
                speech_audio_file = os.path.join(audio_sentences_path, audio_file + '.wav')
                speech_audio_emb_file = speech_audio_file.replace('.wav', '_emb.npy')
                if not os.path.isfile(speech_audio_emb_file):
                    speech_audio_data, sample_rate = librosa.load(
                        os.path.join(audio_sentences_path, audio_file + '.wav'),
                        sr=None)
                    speech_audio_data = resampy.resample(speech_audio_data, sample_rate, 16000)
                    speech_audio_data = audio_processor(speech_audio_data, sampling_rate=16000).input_values[0]

                    speech_audio_embedding = audio_model(speech_audio_data[None, :]).last_hidden_state
                    speech_audio_embedding = np.mean(speech_audio_embedding.numpy(), axis=1).squeeze()
                    np.save(speech_audio_emb_file, speech_audio_embedding)
                else:
                    speech_audio_embedding = np.load(speech_audio_emb_file)

                audio_data_b.append(speech_audio_embedding)

            # Flushing GPU
            del audio_processor
            del audio_model

            audio_data_b = np.array(audio_data_b)
            save_pickle(audio_data_b_save_path, audio_data_b)
        else:
            audio_data_b = load_pickle(audio_data_b_save_path)

        df['audio_features2'] = audio_data_b.tolist()

    if not use_text_features and not use_audio_features:
        raise RuntimeError('Text or audio feature should be enable!')

    Logger.get_logger(__name__).info(f'Running SVM benchmark...')
    Logger.get_logger(__name__).info(f'Combinations: {len(parameters_comb)}')
    Logger.get_logger(__name__).info(f'Space: \n{tuned_parameters}')
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

            cv = PrebuiltCV(n_splits=5, shuffle=True, random_state=42, return_val_indexes=True)
            cv.load_folds(
                os.path.join(ProjectRegistry['prebuilt_folds_dir'], 'm_arg_folds_{:.2f}.json'.format(confidence)))

            all_f1 = []
            cv_val_f1 = []
            cv_test_f1 = []
            cv_ablation_text_f1 = []
            cv_ablation_audio_f1 = []
            for fold_idx, (train_indexes, val_indexes, test_indexes) in enumerate(cv.split(None)):
                train_df = df.iloc[train_indexes]
                val_df = df.iloc[val_indexes]
                test_df = df.iloc[test_indexes]

                train_texts = None
                val_texts = None
                test_texts = None

                train_audio = None
                val_audio = None
                test_audio = None

                x_train = None
                y_train = train_df[label_key].values
                y_train[y_train == 'neither'] = 0
                y_train[y_train == 'support'] = 1
                y_train[y_train == 'attack'] = 2
                y_train = y_train.astype(np.int32)

                x_val = None
                y_val = val_df[label_key].values
                y_val[y_val == 'neither'] = 0
                y_val[y_val == 'support'] = 1
                y_val[y_val == 'attack'] = 2
                y_val = y_val.astype(np.int32)

                x_test = None
                y_test = test_df[label_key].values
                y_test[y_test == 'neither'] = 0
                y_test[y_test == 'support'] = 1
                y_test[y_test == 'attack'] = 2
                y_test = y_test.astype(np.int32)

                if use_text_features:
                    train_a_texts = train_df[text_a_key].values
                    train_b_texts = train_df[text_b_key].values

                    test_a_texts = test_df[text_a_key].values
                    test_b_texts = test_df[text_b_key].values

                    val_a_texts = val_df[text_a_key].values
                    val_b_texts = val_df[text_b_key].values

                    text_vectorizer = TfidfVectorizer()
                    text_vectorizer.fit(np.concatenate((train_a_texts, train_b_texts)))

                    train_a_texts = text_vectorizer.transform(train_a_texts).toarray()
                    train_b_texts = text_vectorizer.transform(train_b_texts).toarray()

                    val_a_texts = text_vectorizer.transform(val_a_texts).toarray()
                    val_b_texts = text_vectorizer.transform(val_b_texts).toarray()

                    test_a_texts = text_vectorizer.transform(test_a_texts).toarray()
                    test_b_texts = text_vectorizer.transform(test_b_texts).toarray()

                    x_train = np.hstack((train_a_texts, train_b_texts))

                    x_val = np.hstack((val_a_texts, val_b_texts))

                    x_test = np.hstack((test_a_texts, test_b_texts))
                else:
                    train_a_texts = None
                    train_b_texts = None

                    val_a_texts = None
                    val_b_texts = None

                    test_a_texts = None
                    test_b_texts = None

                if use_audio_features:
                    train_a_audio = train_df[speech_a_key].values

                    if not use_audio_data:
                        train_a_audio = load_audio_features(features=train_a_audio,
                                                            pooling_sizes=[],
                                                            normalize=False,
                                                            remove_energy=False)
                    train_a_audio = np.array([np.array(item) for item in train_a_audio])

                    train_b_audio = train_df[speech_b_key].values

                    if not use_audio_data:
                        train_b_audio = load_audio_features(features=train_b_audio,
                                                            pooling_sizes=[],
                                                            normalize=False,
                                                            remove_energy=False)
                    train_b_audio = np.array([np.array(item) for item in train_b_audio])

                    test_a_audio = test_df[speech_a_key].values

                    if not use_audio_data:
                        test_a_audio = load_audio_features(features=test_a_audio,
                                                           pooling_sizes=[],
                                                           normalize=False,
                                                           remove_energy=False)
                    test_a_audio = np.array([np.array(item) for item in test_a_audio])

                    test_b_audio = test_df[speech_b_key].values

                    if not use_audio_data:
                        test_b_audio = load_audio_features(features=test_b_audio,
                                                           pooling_sizes=[],
                                                           normalize=False,
                                                           remove_energy=False)
                    test_b_audio = np.array([np.array(item) for item in test_b_audio])

                    val_a_audio = val_df[speech_a_key].values

                    if not use_audio_data:
                        val_a_audio = load_audio_features(features=val_a_audio,
                                                          pooling_sizes=[],
                                                          normalize=False,
                                                          remove_energy=False)
                    val_a_audio = np.array([np.array(item) for item in val_a_audio])

                    val_b_audio = val_df[speech_b_key].values

                    if not use_audio_data:
                        val_b_audio = load_audio_features(features=val_b_audio,
                                                          pooling_sizes=[],
                                                          normalize=False,
                                                          remove_energy=False)
                    val_b_audio = np.array([np.array(item) for item in val_b_audio])

                    if train_a_texts is not None:
                        x_train = np.hstack((train_a_texts, train_b_texts, train_a_audio, train_b_audio))
                        x_test = np.hstack((test_a_texts, test_b_texts, test_a_audio, test_b_audio))
                        x_val = np.hstack((val_a_texts, val_b_texts, val_a_audio, val_b_audio))
                    else:
                        x_train = np.hstack((train_a_audio, train_b_audio))
                        x_test = np.hstack((test_a_audio, test_b_audio))
                        x_val = np.hstack((val_a_audio, val_b_audio))

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

                fold_f1 = f1_score(y_true, y_pred, average='macro', labels=[1, 2])
                all_f1.append(fold_f1)

                if not is_calibrating:
                    fold_val_f1 = f1_score(y_true=y_val, y_pred=clf.predict(svd.transform(x_val)), average='macro',
                                           labels=[1, 2])
                    cv_val_f1.append(fold_val_f1)

                    fold_test_f1 = f1_score(y_true=y_test, y_pred=clf.predict(svd.transform(x_test)), average='macro',
                                            labels=[1, 2])
                    cv_test_f1.append(fold_test_f1)

                    if use_text_features and use_audio_features:
                        # Ablation study

                        # Text
                        fold_test_ablation_text_f1 = f1_score(y_true=y_test,
                                                              y_pred=clf.predict(svd.transform(np.hstack((np.zeros_like(
                                                                  test_a_texts), np.zeros_like(test_b_texts),
                                                                                                          test_a_audio,
                                                                                                          test_b_audio)))),
                                                              average='macro', labels=[1, 2])
                        cv_ablation_text_f1.append(fold_test_ablation_text_f1)

                        # Audio
                        fold_test_ablation_audio_f1 = f1_score(y_true=y_test, y_pred=clf.predict(svd.transform(
                            np.hstack((test_a_texts, test_b_texts, np.zeros_like(test_a_audio),
                                       np.zeros_like(test_b_audio))))), average='macro', labels=[1, 2])
                        cv_ablation_audio_f1.append(fold_test_ablation_audio_f1)

            seed_f1.append(np.mean(all_f1))

            if not is_calibrating:
                val_f1.append(np.mean(cv_val_f1))
                test_f1.append(np.mean(cv_test_f1))

                if use_text_features and use_audio_features:
                    test_ablation_text_f1.append(np.mean(cv_ablation_text_f1))
                    test_ablation_audio_f1.append(np.mean(cv_ablation_audio_f1))

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
        Logger.get_logger(__name__).info(f'Validation F1: {val_f1}')
        Logger.get_logger(__name__).info(f'Test F1: {test_f1}')

        to_save = {
            'is_calibrating': is_calibrating,
            'use_text_features': use_text_features,
            'use_audio_features': use_audio_features,
            'use_audio_data': use_audio_data,
            'validation F1': val_f1,
            'cv validation F1': cv_val_f1,
            'test F1': test_f1,
            'cv test F1': cv_test_f1,
            'tuned_parameters': tuned_parameters
        }

        if use_audio_features and use_text_features:
            Logger.get_logger(__name__).info(f'Test Ablation-Text F1: {test_ablation_text_f1}')
            Logger.get_logger(__name__).info(f'Test Ablation-Audio F1: {test_ablation_audio_f1}')

            to_save['ablation text F1'] = test_ablation_text_f1
            to_save['ablation audio F1'] = test_ablation_audio_f1

        save_json(os.path.join(save_base_path, 'benchmark_info.json'), to_save)
