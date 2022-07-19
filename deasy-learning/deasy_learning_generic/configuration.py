
import abc
from abc import ABC
from collections import Counter
from copy import deepcopy
from typing import AnyStr, Any, List, Callable, Dict, Union

from dataclasses import dataclass
from varname.helpers import Wrapper

from deasy_learning_generic.component import Component
from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag, RegistrationInfo
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.pickle_utils import save_pickle, load_pickle
from deasy_learning_generic.utility.python_utils import get_gridsearch_parameters
from deasy_learning_generic.utility.python_utils import merge
from deasy_learning_generic.utility.search_utils import MatchFilter, EqualityFilter, PassFilter


class Configuration(ABC):
    """
    Base configuration class.
    Configurations might instantiate a component or wrap information for a specific part of a component.
    Configurations are meant to store fields and handle field related verification processes.
    """

    def __init__(self, component_registration_info: RegistrationInfo = None,
                 children: List[RegistrationInfo] = None):
        """
        Instantiates a new Configuration object.

        Args:
            component_registration_info (RegistrationInfo): a RegistrationInfo instance that stores registration
            information about the component associated with the configuration.
            children (List[RegistrationInfo]): a list of RegistrationInfo instances.
        """

        self.component_registration_info = component_registration_info
        children = children if children is not None else []
        self.children = self._build_children(children=children)

        self.conditions = []
        self.excluded_attributes = Wrapper([])  # Utility to access variable's name

        self._exclude_attribute('conditions')
        self._exclude_attribute('framework')
        self._exclude_attribute('component_registration_info')

    def _build_children(self, children: List[RegistrationInfo]) -> List['Configuration']:
        """
        Retrieves registered Configuration objects from the given list of RegistrationInfo instances.

        Args:
            children (List[RegistrationInfo]): a list of RegistrationInfo instances.

        Returns:
            built_children (List[Configuration]): a list of Configuration instances.
        """

        built_children = []
        for child in children:
            built_child = ProjectRegistry.retrieve_configurations_from_info(registration_info=child)
            built_children.append(built_child)

        return built_children

    def add_child(self, child):
        self.children.append(ProjectRegistry.retrieve_configurations_from_info(registration_info=child))

    def _exclude_attribute(self, argument: AnyStr):
        """
        Adds the given attribute name to the list of excluded attribute.

        Args:
            argument (str): an attribute name of current instance.
        """
        self.excluded_attributes.value.append(argument)

    def get_attributes(self) -> Dict:
        """
        Returns a list of current instance attributes excluding those that have been excluded explicitly.
        """
        return {key: value for key, value in vars(self).items()
                if key not in self.excluded_attributes.value and key != self.excluded_attributes.name}

    def add_condition(self, condition: Callable, attribute_names: List[AnyStr], condition_name: AnyStr = None):
        """
        Adds a condition function to be evaluated concerning an attribute subset.

        Args:
            condition (Callable): a function to be called that evaluates an attribute subset.
            attribute_names (List[AnyStr]): a list of attribute names that current Configuration instance has.
            condition_name (AnyStr): the name of the added evaluation function.
        """

        self.conditions.append({
            'condition': condition,
            'argument_names': attribute_names
        })

        # If no name is given, define the name as the condition's position in the list
        if condition_name is not None:
            condition_name = len(self.conditions) - 1

        self.conditions[-1]['condition_name'] = condition_name

    def evaluate_conditions(self):
        """
        Calls each stored condition evaluation function with its corresponding set of arguments.
        """

        for condition_info in self.conditions:
            condition_arguments = condition_info['argument_names']
            retrieved_arguments = []

            # Check if current instance has each attribute
            for arg_name in condition_arguments:
                assert hasattr(self, arg_name)
                retrieved_arguments.append(getattr(self, arg_name))

            if not condition_info['condition'](*retrieved_arguments):
                raise RuntimeError('Condition {0} is not satisfied!'.format(condition_info['condition_name']))

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        """
        Returns the default Configuration instance.

        Returns:
            Configuration instance.
        """
        pass

    def retrieve_component(self, additional_args: Dict = None) -> Component:
        """
        Retrieves the registered Component class associated with the current Configuration instance.

        Returns:
            registered Component class.
        """
        if self.has_component():
            additional_args = additional_args if additional_args is not None else {}
            additional_args = merge(self.get_attributes(), additional_args)
            return ProjectRegistry.retrieve_component_from_info(self.component_registration_info, args=additional_args)

    def _get_configuration_flag(self) -> ComponentFlag:
        """
        Returns the ComponentFlag associated with the current Configuration instance.

        Returns:
            The ComponentFlag associated with the current Configuration instance.
        """
        pass

    def get_configuration_flag(self) -> ComponentFlag:
        """
        Returns the ComponentFlag associated with the current Configuration instance.
        If the current Configuration instance has a component, it returns the component's ComponentFlag.
        """
        if self.has_component():
            return self.component_registration_info.flag
        else:
            return self._get_configuration_flag()

    def has_component(self) -> bool:
        """
        Returns True if the Configuration instance supports a component.
        """
        return True

    def get_flags(self) -> List[ComponentFlag]:
        """
        Returns the total list of ComponentFlag in current Configuration instance.
        The list of ComponentFlag includes the configurations' ComponentFlag and its childrens' one as well.
        """

        flags = [self.get_configuration_flag()] + [child.get_configuration_flag() for child in self.children]
        return flags

    def get_serialization_parameters(self) -> Dict:
        """
        Returns the current Configuration instance fields that can change the data serialization pipeline.

        Returns:
            parameters (dict): {parameter_name: parameter_value} dictionary of current Configuration instance,
            including its children.
        """

        parameters = {}
        if self.children is None:
            return parameters
        else:
            for child in self.children:
                parameters.update(child.get_serialization_parameters())
        return parameters

    def get_component_flag(self) -> Union[ComponentFlag, None]:
        """
        Returns the ComponentFlag of current Configuration instance' component (if any).
        """
        return None

    def get_delta_copy(self, **kwargs) -> 'Configuration':
        """
        Builds a delta copy of current Configuration instance with given input kwargs.

        Args:
            kwargs: attributes that override the existing ones in the current Configuration instance.

        Returns:
            config_copy (Configuration): a copy of current Configuration instance with given override attributes.
        """

        config_copy = deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(config_copy, key):
                # Check if children overwriting
                if key == 'children':
                    value = self._build_children(children=value)

                setattr(config_copy, key, value)
            else:
                Logger.get_logger(__name__).warn(f'{config_copy.__class__.__name__} has no {key} as field! Skipping...')
        return config_copy

    def get_combinations(self, param_dict):
        copies = []
        for param_comb in get_gridsearch_parameters(params_dict=param_dict):
            copies.append((self.get_delta_copy(**param_comb), param_comb))
        return copies

    def register_combinations_from_params(self, param_dict, framework=None, namespace=None, tags=None):
        combinations = self.get_combinations(param_dict=param_dict)
        tags = tags if tags else []

        for comb_config, comb_info in combinations:
            ProjectRegistry.register_configuration(configuration=comb_config,
                                                   framework=framework,
                                                   namespace=namespace,
                                                   tags=tags + [f'{key}={value}' for key, value in comb_info.items()])


class EvaluationConfiguration(Configuration):

    def __init__(self, batch_size=32, verbose=1,
                 epochs=10000, step_checkpoint=None, inference_repetitions=1, **kwargs):
        super(EvaluationConfiguration, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.verbose = verbose
        self.epochs = epochs
        self.step_checkpoint = step_checkpoint
        self.inference_repetitions = inference_repetitions

    @classmethod
    def get_default(cls):
        return EvaluationConfiguration()

    def has_component(self):
        return False

    def get_fit_arguments(self):
        return {
            'epochs': self.epochs,
            'verbose': self.verbose,
            'step_checkpoint': self.step_checkpoint,
            'inference_repetitions': self.inference_repetitions
        }

    def get_inference_arguments(self):
        return {
            'repetitions': self.inference_repetitions
        }

    def _get_configuration_flag(self):
        return ComponentFlag.EVALUATION


class RoutineConfiguration(Configuration):

    def __init__(self, repetitions=1, compute_test_info=True, validation_percentage=None, seeds=None, **kwargs):
        super(RoutineConfiguration, self).__init__(**kwargs)
        self.repetitions = repetitions
        self.compute_test_info = compute_test_info
        self.validation_percentage = validation_percentage
        self.seeds = seeds

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass

    def get_component_flag(self):
        return ComponentFlag.ROUTINE


class CVTestRoutineConfiguration(RoutineConfiguration):

    def __init__(self, split_key, cv_type='kfold', folds_path=None, held_out_key='validation',
                 n_splits=10, shuffle=True, random_state=42, **kwargs):
        super(CVTestRoutineConfiguration, self).__init__(**kwargs)
        self.split_key = split_key
        self.cv_type = cv_type
        self.folds_path = folds_path
        self.held_out_key = held_out_key
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @classmethod
    def get_default(cls) -> 'Configuration':
        return CVTestRoutineConfiguration(split_key=None)


class LooTestRoutineConfiguration(RoutineConfiguration):

    def __init__(self, split_key=None, save_path=None, **kwargs):
        super(LooTestRoutineConfiguration, self).__init__(**kwargs)
        self.split_key = split_key
        self.save_path = save_path

    @classmethod
    def get_default(cls) -> 'Configuration':
        return LooTestRoutineConfiguration()


class HelperConfiguration(Configuration):

    def get_component_flag(self):
        return ComponentFlag.FRAMEWORK_HELPER

    @classmethod
    def get_default(cls) -> 'Configuration':
        return HelperConfiguration()


class CallbackConfiguration(Configuration):

    def get_component_flag(self):
        return ComponentFlag.CALLBACK

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass


class DataLoaderConfiguration(Configuration):

    def __init__(self, label_metrics_map=None, metrics=None, **kwargs):
        super(DataLoaderConfiguration, self).__init__(**kwargs)
        self.label_metrics_map = label_metrics_map
        self.metrics = metrics

    def get_serialization_parameters(self) -> Dict:
        parameters = super(DataLoaderConfiguration, self).get_serialization_parameters()
        parameters['component_registration_info'] = self.component_registration_info
        return parameters

    def get_component_flag(self):
        return ComponentFlag.DATA_LOADER

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass


class ProcessorConfiguration(Configuration):

    def get_component_flag(self):
        return ComponentFlag.PROCESSOR

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass


class ConverterConfiguration(Configuration):

    def __init__(self, feature_registration_info, **kwargs):
        super(ConverterConfiguration, self).__init__(**kwargs)
        self.feature_registration_info = feature_registration_info

    def get_component_flag(self):
        return ComponentFlag.CONVERTER

    def get_serialization_parameters(self) -> Dict:
        return {
            'feature_registration_info': self.feature_registration_info
        }

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass


class CalibratorConfiguration(Configuration):

    def __init__(self, validate_on, validate_condition, max_evaluations=-1, **kwargs):
        super(CalibratorConfiguration, self).__init__(**kwargs)
        self.validate_on = validate_on
        self.validate_condition = validate_condition
        self.max_evaluations = max_evaluations

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass

    @abc.abstractmethod
    def get_search_space(self):
        pass

    def get_configuration_flag(self) -> ComponentFlag:
        return ComponentFlag.CALIBRATOR


class HyperoptCalibratorConfiguration(CalibratorConfiguration):

    def __init__(self, hyperopt_additional_info=None, use_mongo=False,
                 mongo_address='localhost', mongo_port=1234, mongo_dir=None,
                 workers=2, workers_dir=None, poll_interval=0.1, reserve_timeout=10.0,
                 max_consecutive_failures=2, use_subprocesses=False, **kwargs):
        super(HyperoptCalibratorConfiguration, self).__init__(**kwargs)
        self.hyperopt_additional_info = hyperopt_additional_info
        self.use_mongo = use_mongo
        self.mongo_address = mongo_address
        self.mongo_port = mongo_port
        self.mongo_dir = mongo_dir
        self.workers = workers
        self.workers_dir = workers_dir
        self.poll_interval = poll_interval
        self.reserve_timeout = reserve_timeout
        self.max_consecutive_failures = max_consecutive_failures
        self.use_subprocesses = use_subprocesses

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass

    @abc.abstractmethod
    def get_search_space(self):
        pass


@dataclass
class ParamInfo:
    name: AnyStr
    value: Any


class ModelParam(ABC):

    def __init__(self, name, value, flags, allowed_values=None, key=None,
                 disable_value=None, supports_disabling=False):
        self.name = name
        self.value = value
        self.flags = flags
        self.allowed_values = allowed_values
        self.key = key
        self.disable_value = disable_value
        self.supports_disabling = supports_disabling

    def disable(self):
        if self.supports_disabling:
            assert self.disable_value is not None
            self.value = self.disable_value
            Logger.get_logger(__name__).info('Setting parameter {0} to its disabled'
                                             ' value {1} due to triggering constraint'.format(self.name,
                                                                                              self.disable_value))

    def is_value_allowed(self, value):
        if self.allowed_values is not None:
            return value in self.allowed_values
        else:
            return True

    def __repr__(self):
        return 'Name: {0} -- Value: {1} -- Flags: {2}'.format(self.name, self.value, self.flags)

    def get_short_repr(self):
        return 'Name: {0}'.format(self.name)


class ModelParamCondition(ABC):

    def __init__(self, param_names, params_condition, resolution_function):
        self.param_names = param_names
        self.params_condition = params_condition
        self.resolution_function = resolution_function

    def evaluate(self, config):
        params = {name: config[name] for name in self.param_names}

        if not self.params_condition(**params):
            flag, params_to_update = self.resolution_function(**params)
            config.update_from_params_info(params_to_update)
            return flag
        return True


class ModelConfiguration(Configuration):

    def __init__(self, pipeline_configurations, model_params=None, param_conditions=None, **kwargs):
        super(ModelConfiguration, self).__init__(**kwargs)
        self.pipeline_configurations = None
        self._parse_configurations(pipeline_configurations)

        self.model_params = model_params if model_params is not None else {}
        self.param_conditions = param_conditions if param_conditions is not None else []

        self._exclude_attribute('model_params')
        self._exclude_attribute('names')
        self._exclude_attribute('param_conditions')
        self._exclude_attribute('pipeline_registration_info')

        self._evaluate_parameters()

    def __getitem__(self, key):
        return self.model_params[key]

    def __contains__(self, param):
        return param.name in self.model_params

    def _parse_configurations(self, configurations):
        pipeline_configurations = []
        for config_info in configurations:
            assert isinstance(config_info, RegistrationInfo)
            retrieved_config = ProjectRegistry.retrieve_configurations_from_info(registration_info=config_info)
            pipeline_configurations.append(retrieved_config)

        self.pipeline_configurations = pipeline_configurations

    def get_component_flag(self):
        return ComponentFlag.MODEL

    @classmethod
    def get_default(cls):
        raise NotImplementedError()

    def _evaluate_parameters(self):
        names = list(self.model_params.keys())
        if len(set(names)) < len(names):
            raise RuntimeError('Found duplicate parameters! Summary: \n{0}'.format(Counter(names)))

        for condition in self.param_conditions:
            if not condition.evaluate(self):
                raise RuntimeError('A condition has failed! Condition: {}'.format(condition))

    def get_attributes(self):
        pipeline_model_arguments = {}
        for config in self.pipeline_configurations:
            pipeline_model_arguments[config.get_component_flag()] = self.get_flag_parameters(
                config.get_component_flag())
            if config.children is not None and len(config.children):
                for child in config.children:
                    child_flag = child.get_component_flag()
                    pipeline_model_arguments[child_flag] = self.get_flag_parameters(child_flag)

        base_arguments = super(ModelConfiguration, self).get_attributes()
        base_arguments['pipeline_model_arguments'] = pipeline_model_arguments
        return merge(base_arguments, self.get_model_parameters())

    def get_flag_parameters(self, flag, key=None):
        filters = []

        if flag is not None:
            filters.append(MatchFilter(attribute_name='flags', reference_value=[flag]))

        if key is not None:
            filters.append(EqualityFilter(attribute_name='key', reference_value=key))

        # Add
        if flag is None and key is None:
            filters.append(PassFilter(return_value=False))

        filtered_parameters = list(filter(lambda param: all(map(lambda f: f(param), filters)), list(self.model_params.values())))
        return {param.name: param.value for param in filtered_parameters}

    def get_model_parameters(self):
        return self.get_flag_parameters(ComponentFlag.MODEL)

    def add_param(self, model_param):
        # Check if name is already taken
        if model_param.name in self.model_params:
            Logger.get_logger(__name__).info('Could not add parameter {0} with duplicate name!'.format(model_param))
        else:
            self.model_params[model_param.name] = model_param

    def update_param(self, model_param):
        assert model_param.name in self.model_params
        self.model_params[model_param.name] = model_param

    def update_params(self, model_params):
        for param_key, param in model_params.items():
            self.update_param(param)

    def update_from_param_info(self, param_info):
        assert param_info.name in self.model_params, f'{self.__class__.__name__} has no parameter {param_info}!'
        self.model_params[param_info.name].value = param_info.value

    def update_from_params_info(self, params_info):
        for param_info in params_info:
            self.update_from_param_info(param_info=param_info)
        self._evaluate_parameters()

    def add_param_condition(self, param_condition):
        self.param_conditions.append(param_condition)

    def show(self):
        Logger.get_logger(__name__).info('Showing model configuration...')
        Logger.get_logger(__name__).info('Parameters:')
        for param_key, param in self.model_params.items():
            Logger.get_logger(__name__).info('{}'.format(param))

        Logger.get_logger(__name__).info('Conditions:')
        for condition in self.param_conditions:
            Logger.get_logger(__name__).info('{}'.format(condition))

    def to_file(self, filepath):
        save_pickle(filepath, self)

    @staticmethod
    def from_file(filepath, show=False):
        loaded = load_pickle(filepath)
        if show:
            loaded.show()

        return loaded

    def get_serialization_parameters(self):
        parameters = {}
        for config in self.pipeline_configurations:
            config_parameters = config.get_serialization_parameters()
            config_flags = config.get_flags()
            model_config_parameters = {}
            for flag in config_flags:
                model_config_parameters.update(self.get_flag_parameters(flag=flag))
            config_parameters.update(model_config_parameters)
            parameters.update(config_parameters)
        return parameters

    def get_delta_copy(self, **kwargs) -> 'Configuration':
        delta_copy = super(ModelConfiguration, self).get_delta_copy(**kwargs)
        if 'pipeline_configurations' in kwargs:
            delta_copy._parse_configurations(configurations=kwargs['pipeline_configurations'])

        return delta_copy

    def get_delta_param_copy(self, params_info, **kwargs):
        if type(params_info) == dict:
            params_info = [ParamInfo(name=key, value=value) for key, value in params_info.items()]

        delta_copy = self.get_delta_copy(**kwargs)
        delta_copy.update_from_params_info(params_info)
        return delta_copy

    def get_combinations_from_params(self, params_dict, **kwargs):
        copies = []
        for param_comb in get_gridsearch_parameters(params_dict=params_dict):
            params_info = []
            for name, value in param_comb.items():
                params_info.append(ParamInfo(name=name, value=value))
            current_copy = self.get_delta_copy(**kwargs)
            current_copy.update_from_params_info(params_info=params_info)
            copies.append((current_copy, params_info))

        return copies

    def register_combinations_from_params(self, params_dict, framework=None, namespace=None, tags=None, **kwargs):
        combinations = self.get_combinations_from_params(params_dict=params_dict, **kwargs)
        tags = tags if tags else []

        for comb, comb_info in combinations:
            ProjectRegistry.register_configuration(configuration=comb,
                                                   framework=framework,
                                                   namespace=namespace,
                                                   tags=tags + [f'{info.name}={info.value}' for info in comb_info])


class TaskConfiguration(Configuration):
    DEFAULT_KEY = 'default'

    def __init__(self, task_name, reference_task_name=None, **kwargs):
        super(TaskConfiguration, self).__init__(**kwargs)
        self.registrations_info = {}
        self.configurations = {}
        self.task_name = task_name
        self.reference_task_name = reference_task_name if reference_task_name is not None else self.task_name

        self._exclude_attribute('registrations_info')
        self._exclude_attribute('configurations')

    @classmethod
    @abc.abstractmethod
    def get_default(cls) -> 'Configuration':
        pass

    def add_configuration(self, registration_info, key=None, allow_multiple=False, force_update=False):

        if allow_multiple:
            assert key is not None, f"A key is required to store multiple configurations with the same ComponentFlag."
        else:
            if not force_update:
                assert registration_info.flag not in self.registrations_info, 'Trying to add multiple configurations for the same step!'
            key = TaskConfiguration.DEFAULT_KEY

        configuration = ProjectRegistry.retrieve_configurations_from_info(registration_info=registration_info)

        if not force_update:
            try:
                if registration_info.flag in self.registrations_info and \
                        key in self.registrations_info[registration_info.flag]:
                    Logger.get_logger(__name__).info(f'A configuration has already been added with key = {key}. '
                                                     f'Change key or enable force_update for overwriting')
            except KeyError as e:
                raise e

            self.registrations_info.setdefault(registration_info.flag, {}).setdefault(key, registration_info)
            self.configurations.setdefault(registration_info.flag, {}).setdefault(key, configuration)
        else:
            self.registrations_info[registration_info.flag][key] = registration_info
            self.configurations[registration_info.flag][key] = configuration

    def evaluate(self):
        for flag in [ComponentFlag.ROUTINE,
                     ComponentFlag.EVALUATION,
                     ComponentFlag.DATA_LOADER,
                     ComponentFlag.MODEL]:
            assert flag in self.registrations_info, f'Missing configuration for flag {flag}'

        # Compute task name
        assert self.task_name is not None

        if self.reference_task_name is None:
            self.reference_task_name = self.task_name

        # Logger.get_logger(__name__).info(f'Task configuration is valid! '
        #                                  f'Remember to configure a framework helper configuration if not done yet.')

    def get_component_flag(self):
        return ComponentFlag.TASK

    def get_serialization_parameters(self):
        parameters = {}
        for configs in self.configurations.values():
            for config in configs.values():
                # Update config parameters with model override parameters
                config_parameters = config.get_serialization_parameters()
                if config.get_component_flag() != ComponentFlag.MODEL:
                    config_flags = config.get_flags()
                    model_config_parameters = {}
                    for flag in config_flags:
                        model_config_parameters.update(
                            self.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(flag))
                    config_parameters.update(model_config_parameters)
                parameters.update(config_parameters)
        return parameters

    def get_debug_version(self):
        return self
