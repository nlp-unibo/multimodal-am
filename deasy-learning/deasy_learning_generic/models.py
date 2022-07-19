import os
from collections import OrderedDict

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from deasy_learning_generic.composable import Composable
from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.implementations.labels import ClassificationLabel
from deasy_learning_generic.registry import ComponentFlag
from deasy_learning_generic.routines import RoutineSuffixFlag
from deasy_learning_generic.utility.log_utils import Logger

logger = Logger.get_logger(__name__)


class Model(Composable):

    def __init__(self, pipeline_configurations, pipeline_model_arguments, additional_data=None, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.model = None
        self.optimizer = None
        self.additional_data = additional_data

        # Pipeline
        self.pipeline_configurations = pipeline_configurations
        self.pipeline_model_arguments = pipeline_model_arguments
        self.components = OrderedDict()

        # Routine
        self.model_name_additional_suffixes = OrderedDict()

        # Utility
        self.label_parsing_map = {
            'generation': self._parse_generation_output,
            'classification': self._parse_classification_output,
            'regression': self._parse_regression_output
        }

    # Utility

    def _get_additional_info(self):
        return {}

    # Pipeline

    def _add_pipeline_component(self, component, component_name):
        self.components[component_name] = component

    def build_pipeline(self, model_path, is_training=False, filepath=None, routine_suffixes=None):

        Logger.get_logger(__name__).info('Building pipeline...')

        # Extract routine suffixes that influence the pipeline
        routine_suffixes = [routine_suffix for routine_suffix in routine_suffixes if
                            routine_suffix.flag == RoutineSuffixFlag.AFFECTS_PIPELINE]
        routine_prefix = '_'.join(['{0}_{1}'.format(suffix.name, suffix.value) for suffix in routine_suffixes
                                   if suffix.value is not None])

        for pipeline_configuration in self.pipeline_configurations:
            component_registration_info = pipeline_configuration.component_registration_info
            component_flags = pipeline_configuration.get_flags()
            model_component_args = {flag: self.pipeline_model_arguments[flag] for flag in component_flags}

            if component_registration_info.flag in model_component_args:
                component_additional_args = model_component_args[component_registration_info.flag]
                del model_component_args[component_registration_info.flag]
            else:
                component_additional_args = None

            for child in pipeline_configuration.children:
                child_flag = child.component_registration_info.flag
                if child_flag in model_component_args:
                    if component_additional_args is None:
                        component_additional_args = {}
                    component_additional_args.setdefault('children_args', {}).setdefault(child_flag,
                                                                                         model_component_args[
                                                                                             child_flag])
                    del model_component_args[child_flag]

            component = pipeline_configuration.retrieve_component(additional_args=component_additional_args)

            if os.path.isdir(model_path):
                component.load_info(filepath=model_path, prefix=routine_prefix)

            self._add_pipeline_component(component=component,
                                         component_name=component_registration_info.flag)

    def apply_pipeline(self, data, evaluation_config, model_path, suffix, routine_suffixes=None, return_info=False,
                       filepath=None, save_info=False, show_info=False):

        Logger.get_logger(__name__).info(f'[{str(suffix).upper()}] Applying pipeline...')

        # Extract routine suffixes that influence the pipeline
        routine_suffixes = [routine_suffix for routine_suffix in routine_suffixes if
                            routine_suffix.flag == RoutineSuffixFlag.AFFECTS_PIPELINE]
        routine_prefix = '_'.join(['{0}_{1}'.format(suffix.name.value, suffix.value)
                                   for suffix in routine_suffixes if suffix.value is not None])

        if not len(routine_prefix):
            routine_prefix = None

        serialized_component_index = np.where([component.has_data(model_path=model_path, suffix=suffix,
                                                                  save_prefix=routine_prefix,
                                                                  filepath=filepath)
                                               for _, component in self.components.items()])[-1]
        if len(serialized_component_index):
            serialized_component_index = serialized_component_index[-1]
        else:
            serialized_component_index = None

        component_data = data
        pipeline_info = {
            ComponentFlag.EVALUATION: evaluation_config.get_attributes()
        }

        # Skip the pipeline if no data is provided
        if data is None:
            if return_info:
                for component_name, component in self.components.items():
                    if component.children is not None and len(component.children):
                        for child_flag, child in component.children.items():
                            pipeline_info[child_flag] = child.get_info()
                    pipeline_info[component_name] = component.get_info()
                return data, pipeline_info
            else:
                return data

        # Add data information
        pipeline_info[ComponentFlag.DATA_LOADER] = data.get_info()

        for component_index, (component_name, component) in enumerate(self.components.items()):
            component_data, pipeline_info, component_info = component.apply(component_index=component_index,
                                                                            data=component_data, model_path=model_path,
                                                                            suffix=suffix, save_prefix=routine_prefix,
                                                                            pipeline_info=pipeline_info,
                                                                            filepath=filepath,
                                                                            serialized_component_index=serialized_component_index,
                                                                            save_info=save_info, show_info=show_info)
            pipeline_info[component_name] = component_info

        if return_info:
            return component_data, pipeline_info

        return component_data

    # Routine

    def register_model_suffix(self, routine_suffix):
        if routine_suffix.value is not None:
            self.model_name_additional_suffixes[routine_suffix.name] = routine_suffix

    def prepare_for_training(self, train_data):
        pass

    def prepare_for_loading(self, data):
        pass

    def check_after_loading(self):
        pass

    def get_model_name(self):
        name = 'model'
        for key, suffix in self.model_name_additional_suffixes.items():
            name += '_{0}_{1}'.format(key.value, suffix.value)

        return name

    def clear_iteration_status(self):
        self.model_name_additional_suffixes.clear()

    # Inference

    def _parse_generation_output(self, output, model_additional_info=None):
        pass

    def _parse_classification_output(self, output, model_additional_info=None):
        pass

    def _parse_regression_output(self, output, model_additional_info=None):
        pass

    # General

    def set_state(self, model_state):
        pass

    def get_state(self):
        pass

    def _update_internal_state(self, model_additional_info):
        pass

    def save(self, filepath, overwrite=True):
        raise NotImplementedError()

    def load(self, filepath, is_external=False, **kwargs):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()

    def _evaluate_and_predict(self, *args, **kwargs):
        raise NotImplementedError()

    def fit(self, train_data, metrics=None, *args, **kwargs):
        raise NotImplementedError()

    def parse_labels(self, labels):
        return labels

    def parse_predictions(self, raw_predictions, model_additional_info):
        return raw_predictions

    def build_model(self, pipeline_info):
        raise NotImplementedError()


class BaseNetwork(Model):

    # Saving/Weights

    def get_weights(self):
        raise NotImplementedError()

    def set_weights(self, weights):
        raise NotImplementedError()

    # Training/Inference

    def predict(self, data, callbacks=None, suffix=DataSplit.TEST, repetitions=1, metrics=None, is_toy=False):
        raise NotImplementedError()

    def evaluate(self, data, callbacks=None, repetition=1, metrics=None):
        raise NotImplementedError()

    def _evaluate_and_predict(self, data, callbacks=None, suffix=DataSplit.VAL):
        raise NotImplementedError()

    def fit(self, train_data, epochs=1, verbose=1,
            callbacks=None, validation_data=None, step_checkpoint=None,
            metrics=None, inference_repetitions=1):
        raise NotImplementedError()

    def parse_labels(self, labels):
        return labels

    def parse_predictions(self, raw_predictions, model_additional_info):
        return raw_predictions

    # Routine

    # Model definition

    def train_op(self, x, y, additional_info):
        raise NotImplementedError()

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        raise NotImplementedError()


class ClassificationNetwork(BaseNetwork):

    def __init__(self, weight_predictions=True, **kwargs):
        super(ClassificationNetwork, self).__init__(**kwargs)
        self.weight_predictions = weight_predictions

    def compute_output_weights(self, y_train, label_list):
        self.class_weights = {}
        for label in label_list:
            if isinstance(label, ClassificationLabel):
                label_values = y_train[label.name]
                if len(label_values.shape) > 1:
                    label_values = label_values.ravel()
                label_classes = list(range(label.num_values))
                actual_label_classes = list(set(label_values))
                current_weights = compute_class_weight(class_weight='balanced',
                                                       classes=actual_label_classes, y=label_values)
                remaining_classes = set(label_classes).difference(set(actual_label_classes))

                seen_class_weights = {cls: weight for cls, weight in zip(actual_label_classes, current_weights)}

                for remaining in remaining_classes:
                    seen_class_weights[remaining] = 1.0

                self.class_weights.setdefault(label.name, seen_class_weights)

    def _classification_ce(self, targets, logits, label_name, reduce=True):
        raise NotImplementedError()

    def _compute_losses(self, targets, logits, label_list, reduce=True):
        total_loss = None
        loss_info = {}
        for label_idx, label in enumerate(label_list):
            label_targets = targets[label.name]
            label_logits = logits[label.name]

            if label.label_type == 'classification':
                loss = self._classification_ce(targets=label_targets,
                                               logits=label_logits,
                                               label_name=label.name,
                                               reduce=reduce)
            else:
                raise RuntimeError("Invalid label type -> {}".format(label.label_type))

            loss_info.setdefault(label.name, loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_info

    def prepare_for_training(self, train_data):
        self.compute_output_weights(y_train=train_data.get_labels(),
                                    label_list=self.label_list)


class GenerativeNetwork(BaseNetwork):

    def __init__(self, **kwargs):
        super(GenerativeNetwork, self).__init__(**kwargs)
        self.max_generation_length = None

    def _classification_ce(self, targets, logits):
        raise NotImplementedError()

    def _compute_losses(self, targets, logits, label_list):
        total_loss = None
        loss_info = {}
        for label_idx, label in enumerate(label_list):
            label_targets = targets[label.name]
            label_logits = logits[label.name]

            if label.label_type == 'generation':
                loss = self._classification_ce(targets=label_targets,
                                               logits=label_logits)
            else:
                raise RuntimeError("Invalid label type -> {}".format(label.label_type))

            loss_info.setdefault(label.name, loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_info

    def _parse_generation_output(self, output, model_additional_info=None):
        decoded = [self.components[ComponentFlag.TOKENIZER].decode(seq) for seq in output.tolist()]
        return decoded

    def parse_labels(self, labels):
        decoded = {key: [self.components[ComponentFlag.TOKENIZER].decode(seq) for seq in label.tolist()] for key, label
                   in
                   labels.items()}
        return decoded

    def generate(self, x):
        raise NotImplementedError()
