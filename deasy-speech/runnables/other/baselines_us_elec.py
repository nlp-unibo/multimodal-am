import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from deasy_learning_generic.registry import ProjectRegistry
from deasy_learning_generic.utility.log_utils import Logger


def majority_baseline(df, class_distribution):
    majority_class = sorted(class_distribution.items(), key=lambda pair: pair[1], reverse=True)[0][0]

    if task_type == 'asd':
        if majority_class == 'O':
            majority_class = 0
        else:
            majority_class = 1
    else:
        if majority_class == 'Premise':
            majority_class = 0
        else:
            majority_class = 1

    return [majority_class] * df.shape[0]


def random_baseline(df, class_distribution):
    normalization_factor = np.sum(list(class_distribution.values()))
    class_ratios = {key: value / normalization_factor for key, value in class_distribution.items()}

    if task_type == 'asd':
        weights = [class_ratios['O'], class_ratios['Arg']]
        return np.random.choice([0, 1], size=df.shape[0], p=weights, replace=True)
    else:
        weights = [class_ratios['Premise'], class_ratios['Claim']]
        return np.random.choice([0, 1], size=df.shape[0], p=weights, replace=True)


if __name__ == '__main__':
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    ProjectRegistry.set_project_dir(project_dir)

    task_type = 'acd'
    base_path = os.path.join(ProjectRegistry['local_database'], 'mm-uselecdeb60to16')
    df = pd.read_csv(os.path.join(base_path, 'final_dataset.csv'))

    label_key = 'Component'

    baseline_mode = 'random'
    baselines = {
        'majority': majority_baseline,
        'random': random_baseline
    }

    seeds = [15371, 15372, 15373]

    if task_type == 'acd':
        df = df[df[label_key].isin(['Claim', 'Premise'])]
    else:
        df.loc[df[label_key].isin(['Premise', 'Claim']), label_key] = 'Arg'

    class_distribution = df[label_key].value_counts().to_dict()

    train_df = df[df['Set'] == 'TRAIN']
    val_df = df[df['Set'] == 'VALIDATION']
    test_df = df[df['Set'] == 'TEST']

    val_f1 = []
    test_f1 = []
    for seed in seeds:
        np.random.seed(seed=seed)

        val_labels = val_df[label_key].values

        if task_type == 'asd':
            val_labels[val_labels == 'O'] = 0
            val_labels[val_labels == 'Arg'] = 1
        else:
            val_labels[val_labels == 'Premise'] = 0
            val_labels[val_labels == 'Claim'] = 1

        val_labels = val_labels.astype(np.int32)
        val_predictions = baselines[baseline_mode](df=val_df, class_distribution=class_distribution)

        fold_f1 = f1_score(y_true=val_labels, y_pred=val_predictions, average='macro')
        val_f1.append(fold_f1)

        test_labels = test_df[label_key].values

        if task_type == 'asd':
            test_labels[test_labels == 'O'] = 0
            test_labels[test_labels == 'Arg'] = 1
        else:
            test_labels[test_labels == 'Premise'] = 0
            test_labels[test_labels == 'Claim'] = 1

        test_labels = test_labels.astype(np.int32)
        test_predictions = baselines[baseline_mode](df=test_df, class_distribution=class_distribution)

        fold_f1 = f1_score(y_true=test_labels, y_pred=test_predictions, average='macro')
        test_f1.append(fold_f1)

    Logger.get_logger(__name__).info(f'Baseline mode = {baseline_mode} Val: {np.mean(val_f1)}')
    Logger.get_logger(__name__).info(f'Baseline mode = {baseline_mode} Test: {np.mean(test_f1)}')
