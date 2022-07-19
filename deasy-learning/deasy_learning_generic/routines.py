
import os

import numpy as np
from abc import ABC

from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.composable import Composable
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.routine_utils import PrebuiltCV, PrebuiltLOO, RoutineSuffix, RoutineSuffixFlag, \
    RoutineSuffixType, RoutineStatistics


class Routine(Composable, ABC):

    def __init__(self, framework_helper, data_loader, test_path, model_config, evaluation_config,
                 model_path, train_test_path=None, train_model_path=None, callbacks=None, repetitions=1,
                 compute_test_info=True, validation_percentage=None, seeds=None, **kwargs):
        super(Routine, self).__init__(**kwargs)
        self.framework_helper = framework_helper
        self.data_loader = data_loader
        self.test_path = test_path
        self.train_test_path = train_test_path if train_test_path is not None else self.test_path
        self.model_config = model_config
        self.evaluation_config = evaluation_config
        self.model_path = model_path
        self.train_model_path = train_model_path if train_model_path is not None else self.model_path
        self.callbacks = callbacks if callbacks is not None else []
        self.repetitions = repetitions
        self.compute_test_info = compute_test_info
        self.validation_percentage = validation_percentage

        if seeds is not None:
            assert len(seeds) == repetitions, f'Inconsistent number of seeds given for specified repetitions!' \
                                              f'Got {len(seeds)} seeds for {repetitions} repetitions.'
        else:
            seeds = np.random.randint(low=1, high=10000, size=repetitions)

        self.seeds = seeds

    def build_routine_splits(self, train_data=None, val_data=None, test_data=None, is_training=False, **kwargs):
        raise NotImplementedError()

    def _execute(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare(self):
        pass

    def train(self, save_results=False):
        raise NotImplementedError()

    def forward(self, save_results=False):
        raise NotImplementedError()


class TrainAndTestRoutine(Routine):

    def _get_random_split(self, data):
        amount = int(len(data) * self.validation_percentage)
        all_indexes = np.arange(len(data))
        split_indexes = np.random.choice(all_indexes, size=amount, replace=False)
        remaining_indexes = np.array([idx for idx in all_indexes if idx not in split_indexes])
        split_data = data[split_indexes]
        remaining_data = data[remaining_indexes]

        return remaining_data, split_data

    def get_validation_split(self, data):
        return self._get_random_split(data=data)

    def build_routine_splits(self, train_data=None, val_data=None, test_data=None, is_training=False, **kwargs):

        if is_training:
            assert train_data is not None

        if train_data is not None and val_data is None:
            assert self.validation_percentage is not None, "Routine is expected to build the validation data," \
                                                           " but no validation percentage was given"

            train_data, val_data = self.get_validation_split(data=train_data)

        return train_data, val_data, test_data

    def _execute(self, train_data, val_data, test_data, repetition, metrics, statistics,
                 routine_suffixes=None, is_training=False, save_results=False):
        # Load model
        model = self.model_config.retrieve_component()

        # Apply model pipeline to each data split
        routine_suffixes = routine_suffixes if routine_suffixes is not None else []
        routine_suffixes.append(RoutineSuffix(name=RoutineSuffixType.REPETITION,
                                              value=str(repetition),
                                              flag=RoutineSuffixFlag.NO_EFFECT))
        for routine_suffix in routine_suffixes:
            model.register_model_suffix(routine_suffix=routine_suffix)

        model.build_pipeline(model_path=self.model_path,
                             is_training=is_training, filepath=self.train_test_path, routine_suffixes=routine_suffixes)

        conv_train_data, pipeline_info = model.apply_pipeline(data=train_data, evaluation_config=self.evaluation_config,
                                                              model_path=self.model_path, suffix=DataSplit.TRAIN,
                                                              routine_suffixes=routine_suffixes, return_info=True,
                                                              filepath=self.test_path)

        conv_val_data = model.apply_pipeline(data=val_data, evaluation_config=self.evaluation_config,
                                             model_path=self.model_path, suffix=DataSplit.VAL,
                                             routine_suffixes=routine_suffixes,
                                             filepath=self.test_path)

        conv_test_data = model.apply_pipeline(data=test_data, evaluation_config=self.evaluation_config,
                                              model_path=self.model_path,
                                              suffix=DataSplit.TEST, routine_suffixes=routine_suffixes,
                                              save_info=True,
                                              show_info=True,
                                              filepath=self.test_path)

        # Building model

        # Custom callbacks only
        for callback in self.callbacks:
            callback.on_build_model_begin(logs={'model': model})

        model.build_model(pipeline_info=pipeline_info)

        # Custom callbacks only
        for callback in self.callbacks:
            callback.on_build_model_end(logs={'model': model})

        if is_training:
            # Training
            model.prepare_for_training(train_data=conv_train_data)

            model.fit(train_data=conv_train_data, callbacks=self.callbacks, validation_data=conv_val_data,
                      metrics=metrics, **self.evaluation_config.get_fit_arguments())
        else:
            model.prepare_for_loading(data=conv_test_data if conv_test_data is not None else conv_val_data)

            for callback in self.callbacks:
                callback.on_model_load_begin(logs={'model': model})

            model.load(filepath=os.path.join(self.train_test_path, model.get_model_name()))

            for callback in self.callbacks:
                callback.on_model_load_end(logs={'model': model})

            model.check_after_loading()

        # Inference
        if val_data is not None:
            val_predictions, val_metrics = model.predict(data=conv_val_data,
                                                         callbacks=self.callbacks,
                                                         suffix=DataSplit.VAL,
                                                         metrics=metrics,
                                                         **self.evaluation_config.get_inference_arguments())

            if val_metrics:
                statistics.add_or_update_statistics(statistics=val_metrics,
                                                    suffix=DataSplit.VAL,
                                                    repetition=repetition,
                                                    routine_suffixes=routine_suffixes)

        if self.compute_test_info and test_data is not None:
            test_predictions, test_metrics = model.predict(data=conv_test_data,
                                                           callbacks=self.callbacks,
                                                           suffix=DataSplit.TEST,
                                                           metrics=metrics,
                                                           **self.evaluation_config.get_inference_arguments())

            statistics.add_or_update_predictions(predictions=test_predictions,
                                                 suffix=DataSplit.TEST,
                                                 repetition=repetition,
                                                 routine_suffixes=routine_suffixes)

            if test_metrics:
                statistics.add_or_update_statistics(statistics=test_metrics,
                                                    suffix=DataSplit.TEST,
                                                    repetition=repetition,
                                                    routine_suffixes=routine_suffixes)

        # Save model
        if save_results:
            model_save_path = os.path.join(self.train_test_path, model.get_model_name())
            model.save(filepath=model_save_path)

        # Flush
        self.framework_helper.clear_session()
        model.clear_iteration_status()

        return statistics

    def _run(self, is_training=False, save_results=False):
        metrics = self.data_loader.get_metrics()
        self.data_loader.build_data_splits()

        statistics = RoutineStatistics(repetitions=self.repetitions)

        for repetition in range(self.repetitions):
            Logger.get_logger(__name__).info('Repetition {0}/{1}'.format(repetition + 1, self.repetitions))

            # Seeds
            repetition_seed = self.seeds[repetition]
            self.framework_helper.set_seed(seed=repetition_seed)

            # Get data splits
            train_data, val_data, test_data = self.data_loader.get_data_splits()
            train_data, val_data, test_data = self.build_routine_splits(train_data=train_data,
                                                                        val_data=val_data,
                                                                        test_data=test_data,
                                                                        is_training=is_training)

            statistics = self._execute(train_data=train_data,
                                       val_data=val_data,
                                       test_data=test_data,
                                       repetition=repetition,
                                       metrics=metrics,
                                       is_training=is_training,
                                       save_results=save_results,
                                       statistics=statistics)

        statistics.average()
        statistics.display()

        if save_results:
            statistics.save(filepath=self.test_path)

        return statistics

    def train(self, save_results=False):
        return self._run(is_training=True, save_results=save_results)

    def forward(self, save_results=False):
        return self._run(is_training=False, save_results=save_results)


class CVTestRoutine(TrainAndTestRoutine):

    def __init__(self, split_key, cv_type='kfold', folds_path=None, held_out_key='validation',
                 n_splits=10, shuffle=True, random_state=42, **kwargs):
        super(CVTestRoutine, self).__init__(**kwargs)
        self.split_key = split_key
        self.cv_type = cv_type
        self.folds_path = folds_path
        self.held_out_key = held_out_key

        self.cv = PrebuiltCV(cv_type=self.cv_type, held_out_key=held_out_key, folds_path=folds_path,
                             n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def build_routine_splits(self, train_data=None, val_data=None, test_data=None, is_training=False, test_indexes=None,
                             key=None, val_indexes=None):

        if is_training:
            assert key is not None
            assert test_indexes is not None or val_indexes is not None
            assert train_data is not None

        # Here, train_data might contain: (train, val) or (train, test) according to split keys
        # If (train, test) -> we need to build the validation data

        if train_data is not None:

            # Test data is given -> we use fold keys to define the validation data
            if test_data is not None:
                val_data = train_data[train_data.belongs_to(train_data[key], val_indexes)]
                train_data = train_data[np.logical_not(train_data.belongs_to(train_data[key], val_indexes))]

            # Test data must be built from fold keys
            else:
                test_data = train_data[train_data.belongs_to(train_data[key], test_indexes)]
                train_data = train_data[~train_data.belongs_to(train_data[key], test_indexes)]

                # We then build the validation data

                # We apply a simple split
                if val_data is None:
                    if val_indexes is None:
                        assert self.validation_percentage is not None
                        train_data, val_data = self.get_validation_split(data=train_data)

                    # The CV split also provides validation indexes
                    else:
                        val_data = train_data[train_data.belongs_to(train_data[key], val_indexes)]
                        train_data = train_data[~train_data.belongs_to(train_data[key], val_indexes)]

        return train_data, val_data, test_data

    def _run(self, is_training=False, save_results=False):
        metrics = self.data_loader.get_metrics()
        self.data_loader.build_data_splits()

        statistics = RoutineStatistics(repetitions=self.repetitions)

        if is_training:
            assert self.data_loader[DataSplit.TRAIN] is not None
            self.cv.build_folds(data=self.data_loader[DataSplit.TRAIN], split_key=self.split_key)
        else:
            assert self.folds_path is not None
            self.cv.load_folds(load_path=self.folds_path)

        for repetition in range(self.repetitions):
            Logger.get_logger(__name__).info('Repetition {0}/{1}'.format(repetition + 1, self.repetitions))

            # Seed
            repetition_seed = self.seeds[repetition]
            self.framework_helper.set_seed(seed=repetition_seed)

            for fold_idx, (train_indexes, val_indexes, test_indexes) in enumerate(self.cv.split(None)):
                Logger.get_logger(__name__).info(
                    'Starting Fold {0}/{1}'.format(fold_idx + 1, self.cv.n_splits))

                # Get data splits
                train_data, val_data, test_data = self.data_loader.get_data_splits()
                train_data, val_data, test_data = self.build_routine_splits(train_data=train_data,
                                                                            val_data=val_data,
                                                                            test_data=test_data,
                                                                            is_training=is_training,
                                                                            test_indexes=test_indexes,
                                                                            key=self.split_key,
                                                                            val_indexes=val_indexes)

                statistics = self._execute(train_data=train_data,
                                           val_data=val_data,
                                           test_data=test_data,
                                           repetition=repetition,
                                           metrics=metrics,
                                           is_training=is_training,
                                           routine_suffixes=[
                                               RoutineSuffix(
                                                   name=RoutineSuffixType.FOLD,
                                                   value=str(fold_idx),
                                                   flag=RoutineSuffixFlag.AFFECTS_PIPELINE)
                                           ],
                                           save_results=save_results,
                                           statistics=statistics)

        statistics.average()
        statistics.display()

        if save_results:
            statistics.save(filepath=self.test_path)

        return statistics


class LooTestRoutine(TrainAndTestRoutine):

    def __init__(self, split_key=None, save_path=None, **kwargs):
        super(LooTestRoutine, self).__init__(**kwargs)
        self.split_key = split_key

        self.loo = PrebuiltLOO(save_path=save_path)

    def _get_split_values(self, train_data, split_key):
        return np.unique(train_data[split_key].values)

    def build_routine_splits(self, train_data=None, val_data=None, test_data=None, is_training=False,
                             key_values=None, key=None):

        if is_training:
            assert train_data is not None
            assert key is not None
            assert key_values is not None

        if train_data is not None:
            # We have validation data -> we build test data according to split keys
            if val_data is not None:
                test_data = train_data[train_data.belongs_to(train_data[key], key_values)]
                train_data = train_data[~train_data.belongs_to(train_data[key], key_values)]

            # No validation data
            else:
                # If we don't have test data -> we build it according to split key
                if test_data is None:
                    test_data = train_data[train_data.belongs_to(train_data[key], key_values)]
                    train_data = train_data[~train_data.belongs_to(train_data[key], key_values)]

                    # Then we have to build the validation data
                    assert self.validation_percentage is not None
                    train_data, val_data = self.get_validation_split(data=train_data)

                # We might not have test data (this is fine, if not explicitly requested)
                else:
                    val_data = train_data[train_data.belongs_to(train_data[key], key_values)]
                    train_data = train_data[~train_data.belongs_to(train_data[key], key_values)]

        return train_data, val_data, test_data

    def _run(self, is_training=False, save_results=False):
        metrics = self.data_loader.get_metrics()
        self.data_loader.build_data_splits()

        statistics = RoutineStatistics(repetitions=self.repetitions)

        if is_training:
            assert self.data_loader[DataSplit.TRAIN] is not None
            self.loo.build_split_values(train_data=self.data_loader[DataSplit.TRAIN],
                                        split_key=self.split_key)
            self.loo.save_split_values()
        else:
            self.loo.load_split_values()

        split_values = self.loo.split_values

        for repetition in range(self.repetitions):
            Logger.get_logger(__name__).info('Repetition {0}/{1}'.format(repetition + 1, self.repetitions))

            # Seed
            repetition_seed = self.seeds[repetition]
            self.framework_helper.set_seed(seed=repetition_seed)

            for train_keys_indexes, excluded_key_indexes in self.loo.split(split_values):
                excluded_key = split_values[excluded_key_indexes]

                Logger.get_logger(__name__).info('Excluding: {}'.format(excluded_key))

                # Get data splits
                train_data, val_data, test_data = self.data_loader.get_data_splits()
                train_data, val_data, test_data = self.build_routine_splits(train_data=train_data,
                                                                            val_data=val_data,
                                                                            test_data=test_data,
                                                                            is_training=is_training,
                                                                            key_values=excluded_key,
                                                                            key=self.split_key)

                statistics = self._execute(train_data=train_data,
                                           val_data=val_data,
                                           test_data=test_data,
                                           repetition=repetition,
                                           metrics=metrics,
                                           is_training=is_training,
                                           routine_suffixes=[
                                               RoutineSuffix(
                                                   name=RoutineSuffixType.LOO,
                                                   value=str(excluded_key),
                                                   flag=RoutineSuffixFlag.AFFECTS_PIPELINE)
                                           ],
                                           save_results=save_results,
                                           statistics=statistics)

        statistics.average()
        statistics.display()

        if save_results:
            statistics.save(filepath=self.test_path)

        return statistics
