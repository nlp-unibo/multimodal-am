import numpy as np
import os
from deasy_learning_generic.registry import ProjectRegistry
from deasy_learning_generic.utility.log_utils import Logger
import pandas as pd
from deasy_learning_generic.utility.routine_utils import PrebuiltCV
from sklearn.metrics import f1_score


def majority_baseline(df, class_distribution):
    majority_class = sorted(class_distribution.items(), key=lambda pair: pair[1], reverse=True)[0][0]
    if majority_class == 'neither':
        majority_class = 0
    elif majority_class == 'support':
        majority_class = 1
    else:
        majority_class = 2

    return [majority_class] * df.shape[0]


def random_baseline(df, class_distribution):
    normalization_factor = np.sum(list(class_distribution.values()))
    class_ratios = {key: value / normalization_factor for key, value in class_distribution.items()}
    weights = [class_ratios['neither'], class_ratios['support'], class_ratios['attack']]
    return np.random.choice([0, 1, 2], size=df.shape[0], p=weights, replace=True)


if __name__ == '__main__':
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    ProjectRegistry.set_project_dir(project_dir)

    confidence = 0.85
    base_path = os.path.join(ProjectRegistry['local_database'], 'm-arg')
    df = pd.read_csv(os.path.join(base_path, 'final_dataset_{:.2f}.csv'.format(confidence)))

    cv = PrebuiltCV(n_splits=5, shuffle=True, random_state=42, return_val_indexes=True)
    cv.load_folds(os.path.join(ProjectRegistry['prebuilt_folds_dir'], 'm_arg_folds_{:.2f}.json'.format(confidence)))

    label_key = 'relation'
    class_distribution = df[label_key].value_counts().to_dict()

    baseline_mode = 'random'
    baselines = {
        'majority': majority_baseline,
        'random': random_baseline
    }

    seeds = [15371, 15372, 15373]

    val_f1 = []
    test_f1 = []
    for seed in seeds:
        np.random.seed(seed=seed)

        fold_val_f1 = []
        fold_test_f1 = []
        for fold_idx, (train_indexes, val_indexes, test_indexes) in enumerate(cv.split(None)):
            train_df = df.iloc[train_indexes]
            val_df = df.iloc[val_indexes]
            test_df = df.iloc[test_indexes]

            val_labels = val_df[label_key].values
            val_labels[val_labels == 'neither'] = 0
            val_labels[val_labels == 'support'] = 1
            val_labels[val_labels == 'attack'] = 2
            val_labels = val_labels.astype(np.int32)
            val_predictions = baselines[baseline_mode](df=val_df, class_distribution=class_distribution)

            fold_f1 = f1_score(y_true=val_labels, y_pred=val_predictions, average='macro', labels=[1, 2])
            fold_val_f1.append(fold_f1)

            test_labels = test_df[label_key].values
            test_labels[test_labels == 'neither'] = 0
            test_labels[test_labels == 'support'] = 1
            test_labels[test_labels == 'attack'] = 2
            test_labels = test_labels.astype(np.int32)
            test_predictions = baselines[baseline_mode](df=test_df, class_distribution=class_distribution)

            fold_f1 = f1_score(y_true=test_labels, y_pred=test_predictions, average='macro', labels=[1, 2])
            fold_test_f1.append(fold_f1)

        val_f1.append(np.mean(fold_val_f1))
        test_f1.append(np.mean(fold_test_f1))

    Logger.get_logger(__name__).info(f'Baseline mode = {baseline_mode} Val: {np.mean(val_f1)}')
    Logger.get_logger(__name__).info(f'Baseline mode = {baseline_mode} Test: {np.mean(test_f1)}')
