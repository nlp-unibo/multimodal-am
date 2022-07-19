

import os
from enum import Enum
from typing import AnyStr

import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from sklearn.model_selection._split import _BaseKFold

from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.utility.json_utils import save_json, load_json
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.pickle_utils import save_pickle
from deasy_learning_generic.utility.printing_utils import prettify_statistics, prettify_value


class RoutineSuffixFlag(Enum):
    AFFECTS_PIPELINE = 'affects_pipeline'
    NO_EFFECT = 'no_effect'

    def __str__(self):
        return str(self.value)


class RoutineSuffixType(Enum):
    REPETITION = 'repetition'
    FOLD = 'fold'
    LOO = 'loo'

    def __str__(self):
        return str(self.value)


@dataclass
class RoutineSuffix:
    name: RoutineSuffixType
    value: AnyStr
    flag: RoutineSuffixFlag


class StatisticsNode(object):

    def __init__(self, name, parent=None, data=None, store_operations=False):
        self.name = name
        self.parent = parent
        self.data = data
        self.children = {}
        self.store_operations = store_operations

    def add_child(self, child):
        self.children.setdefault(child.name, child)

    def get_child(self, name):
        return self.children[name]

    def average(self):
        # Average of leaf is its value
        if not len(self.children):
            average = self.data
        else:
            try:
                average = np.mean([child.average() for child_name, child in self.children.items()], axis=0)
            except TypeError as e:
                Logger.get_logger(__name__).info(f'Could not compute average. Reason: {e}')
                average = None
        
        # Store average only if enabled
        if self.store_operations:
            self.data = average

        return average

    def show(self, depth=1):
        to_string = f'{self.name}'
        if self.data is not None:
            to_string += f' --> {prettify_value(self.data)}'

        spacing = ['\t'] * depth
        spacing = ''.join(spacing)
        for child_name, child in self.children.items():
            to_string += f'\n{spacing} --> {child.show(depth=depth + 1)}'

        return to_string

    def to_dict(self):
        current_dict = {self.name: {}}
        if self.data is not None:
            current_dict[self.name]['data'] = self.data
        if len(self.children):
            current_dict[self.name]['children'] = [child.to_dict() for child_name, child in self.children.items()]

        return current_dict


class RoutineStatistics(object):

    def __init__(self, repetitions=None):
        self.repetitions = repetitions
        self.statistics = StatisticsNode(name='root')
        self.predictions = StatisticsNode(name='root')
        self.averaged = False

    def _navigate_and_add_node(self, root, key_path, name=None, data=None, store_operations=False):
        current_node = root

        for key in key_path:
            try:
                current_node = current_node.get_child(key)
            except KeyError:
                new_node = StatisticsNode(name=key, parent=current_node, store_operations=current_node.store_operations)
                current_node.add_child(new_node)
                current_node = new_node

        if name is not None:
            current_node.add_child(StatisticsNode(name=name, data=data, store_operations=store_operations))

        return current_node

    def _determine_access_keys(self, routine_suffixes=None):
        access_keys = []
        if routine_suffixes is not None:
            for routine_suffix in routine_suffixes:
                if routine_suffix.name in [RoutineSuffixType.FOLD, RoutineSuffixType.LOO]:
                    access_keys.append(f'{routine_suffix.name}_{routine_suffix.value}')

        return access_keys

    def get_data(self, key_path):
        return self._navigate_and_add_node(root=self.statistics, key_path=key_path).data

    def add_or_update_statistics(self, statistics, repetition, suffix=DataSplit.VAL,
                                 routine_suffixes=None, verbose=False):
        access_keys = self._determine_access_keys(routine_suffixes=routine_suffixes)

        # Navigate tree
        current_node = self._navigate_and_add_node(root=self.statistics, key_path=[suffix.value])

        for key, value in statistics.items():

            if len(access_keys):
                key_path = [key, f'repetition_{repetition}'] + access_keys[:-1]
                name = access_keys[-1]
            else:
                key_path = [key, f'repetition_{repetition}']
                name = f'repetition_{repetition}'

            current_node.add_child(StatisticsNode(name=key, parent=current_node, store_operations=True))
            self._navigate_and_add_node(root=current_node, key_path=key_path,
                                        name=name,
                                        data=value, store_operations=True)

        if verbose:
            Logger.get_logger(__name__).info(
                f'[Split = {suffix.value}] Update step statistics: {prettify_statistics(statistics)}')

    def add_or_update_predictions(self, predictions, repetition, suffix=DataSplit.VAL, routine_suffixes=None,
                                  verbose=False):
        access_keys = self._determine_access_keys(routine_suffixes=routine_suffixes)

        # Navigate dict data
        current_node = self._navigate_and_add_node(root=self.predictions,
                                                   key_path=[suffix.value, f'repetition_{repetition}'] + access_keys[:-1])
        name = access_keys[-1] if len(access_keys) > 0 else 'predictions'
        current_node.add_child(StatisticsNode(name=name, data=predictions, parent=current_node))

        if verbose:
            Logger.get_logger(__name__).info(f'[Split = {suffix.value}] Update step predictions size: {len(predictions)}')

    def average(self):
        if not self.averaged:
            Logger.get_logger(__name__).info(f'Computing average statistics...')
            self.statistics.average()
            self.averaged = True

    def display(self):
        Logger.get_logger(__name__).info(f'Displaying statistics...\n{self.statistics.show()}')

    def save(self, filepath):
        save_json(os.path.join(filepath, 'statistics.json'), self.statistics.to_dict())
        save_json(os.path.join(filepath, 'predictions.json'), self.predictions.to_dict())

        save_pickle(os.path.join(filepath, 'statistics.pickle'), self.statistics)
        save_pickle(os.path.join(filepath, 'predictions.pickle'), self.predictions)

    def reset(self):
        self.statistics = StatisticsNode(name='root')
        self.predictions = StatisticsNode(name='root')
        self.averaged = False



class PrebuiltCV(_BaseKFold):
    """
    Simple CV wrapper for custom fold definition.
    """

    def __init__(self, folds_path=None, return_val_indexes=True,
                 cv_type='kfold', held_out_key='validation', **kwargs):
        super(PrebuiltCV, self).__init__(**kwargs)
        self.folds = None
        self.return_val_indexes = return_val_indexes
        self.key_listing = None
        self.folds_path = folds_path
        self.held_out_key = held_out_key

        if cv_type == 'kfold':
            self.cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        elif cv_type == 'stratifiedkfold':
            self.cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle)
        else:
            raise AttributeError('Invalid cv_type! Got: {}'.format(cv_type))

    def _build_folds(self, data, split_key):
        data_labels = data[split_key].values
        self.folds = {}
        for fold, (train_indexes, held_out_indexes) in enumerate(self.cv.split(data, data_labels)):
            self.folds['fold_{}'.format(fold)] = {
                'train': train_indexes,
                self.held_out_key: held_out_indexes
            }

    def build_folds(self, data, split_key):
        if self.folds_path is not None:
            self.load_folds(load_path=self.folds_path)
        else:
            self._build_folds(data=data, split_key=split_key)

    def build_all_sets_folds(self, X, y, validation_n_splits=None):
        assert self.held_out_key == 'test'

        validation_n_splits = self.n_splits if validation_n_splits is None else validation_n_splits

        self.folds = {}
        for fold, (train_indexes, held_out_indexes) in enumerate(self.cv.split(X, y)):
            sub_X = X[train_indexes]
            sub_y = y[train_indexes]

            self.cv.n_splits = validation_n_splits
            sub_train_indexes, sub_val_indexes = list(self.cv.split(sub_X, sub_y))[0]
            self.cv.n_splits = self.n_splits

            self.folds['fold_{}'.format(fold)] = {
                'train': train_indexes[sub_train_indexes],
                self.held_out_key: held_out_indexes,
                'validation': train_indexes[sub_val_indexes]
            }

    def load_dataset_list(self, load_path):
        return load_json(load_path)

    def save_folds(self, save_path, tolist=False):
        if tolist:
            to_save = {}
            for fold_key in self.folds:
                for split_set in self.folds[fold_key]:
                    to_save.setdefault(fold_key, {}).setdefault(split_set, self.folds[fold_key][split_set].tolist())
            save_json(save_path, to_save)
        else:
            save_json(save_path, self.folds)

    def load_folds(self, load_path):
        self.folds = load_json(load_path)
        self.n_splits = len(self.folds)
        key_path = load_path.split('.json')[0] + '_listing.json'
        self.key_listing = load_json(key_path)
        self.key_listing = np.array(self.key_listing)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            yield self.folds[fold][self.held_out_key]

    def split(self, X, y=None, groups=None):
        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            val_indexes = self.key_listing[self.folds[fold]['validation']] if 'validation' in self.folds[fold] else None
            test_indexes = self.key_listing[self.folds[fold]['test']] if 'test' in self.folds[fold] else None

            assert val_indexes is not None or test_indexes is not None

            train_indexes = self.key_listing[self.folds[fold]['train']]

            if self.return_val_indexes:
                yield train_indexes, val_indexes, test_indexes
            else:
                yield train_indexes, test_indexes


class PrebuiltLOO(LeaveOneOut):

    def __init__(self, save_path=None, **kwargs):
        super(PrebuiltLOO, self).__init__(**kwargs)
        self.save_path = save_path
        self.split_values = None

    def build_split_values(self, train_data, split_key):
        self.split_values = np.unique(train_data[split_key].values)

    def save_split_values(self):
        assert self.save_path is not None
        save_json(self.save_path, self.split_values)

    def load_split_values(self):
        assert self.save_path is not None
        self.split_values = load_json(self.save_path)
