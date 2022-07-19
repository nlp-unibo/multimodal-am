import ast
import importlib
import os
from enum import Enum, EnumMeta

from aenum import extend_enum

from deasy_learning_generic.utility.json_utils import load_json
from deasy_learning_generic.utility.search_utils import PassFilter, parse_and_create_filter


class ComponentFlagMeta(EnumMeta):

    def __contains__(cls, item):
        return item in cls.__members__.values()


class ComponentFlag(str, Enum, metaclass=ComponentFlagMeta):
    DATA_LOADER = "data_loader"
    PROCESSOR = "processor"
    TOKENIZER = "tokenizer"
    CONVERTER = "converter"
    MODEL = "model"
    CALLBACK = 'callback'
    FEATURE = 'feature'
    METRIC = 'metric'
    LABEL = "label"
    FRAMEWORK_HELPER = "framework_helper"
    ROUTINE = 'routine'
    EVALUATION = 'evaluation'
    CALIBRATOR = 'calibrator'
    TASK = 'task'

    @staticmethod
    def add_component_flag(name, value):
        if name not in ComponentFlag.__members__:
            extend_enum(ComponentFlag, name, value)

    def __str__(self):
        return str(self.value)


class MetaProjectRegistry(type):

    def __getitem__(cls, key):
        return cls.CONSTANTS[key]

    def __setitem__(cls, key, value):
        cls.CONSTANTS[key] = value


class RegisteredComponent(object):

    def __init__(self, class_type, constructor=None):
        self.class_type = class_type
        self.constructor = constructor


class RegisteredConfiguration(object):

    def __init__(self, config):
        self.config = config


class RegistrationInfo(object):
    _ATTRIBUTE_SEPARATOR = '--'
    _KEY_VALUE_SEPARATOR = ':'

    def __init__(self, flag, framework, internal_key, tags=None, namespace=None):
        self.internal_key = internal_key
        self.flag = flag
        self.framework = framework
        self.tags = sorted(list(set(tags))) if tags is not None else tags
        self.namespace = namespace

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        to_return = f'internal_key:{self.internal_key}' \
                    f'{RegistrationInfo._ATTRIBUTE_SEPARATOR}flag{RegistrationInfo._KEY_VALUE_SEPARATOR}{self.flag}' \
                    f'{RegistrationInfo._ATTRIBUTE_SEPARATOR}framework{RegistrationInfo._KEY_VALUE_SEPARATOR}{self.framework}'

        if self.tags is not None:
            to_return += f'{RegistrationInfo._ATTRIBUTE_SEPARATOR}tags{RegistrationInfo._KEY_VALUE_SEPARATOR}{self.tags}'
        if self.namespace is not None:
            to_return += f'{RegistrationInfo._ATTRIBUTE_SEPARATOR}namespace{RegistrationInfo._KEY_VALUE_SEPARATOR}{self.namespace}'

        return to_return

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        default_condition = lambda other: self.internal_key == other.internal_key and \
                                          self.flag == other.flag and \
                                          self.framework == other.framework

        # Equality between tags is possible since we sort tags in __init__
        tags_condition = lambda other: (self.tags is not None and other.tags is not None and self.tags == other.tags) \
                                       or (self.tags is None and other.tags is None)

        namespace_condition = lambda other: (self.namespace is not None
                                             and other.namespace is not None
                                             and self.namespace == other.namespace) \
                                            or (self.namespace is None and other.namespace is None)

        return default_condition(other) \
               and tags_condition(other) \
               and namespace_condition(other)

    def partial_match(self, other):
        default_condition = lambda other: self.internal_key == other.internal_key and \
                                          self.flag == other.flag and \
                                          self.framework == other.framework

        tags_condition = lambda other: (self.tags is not None
                                        and other.tags is not None
                                        and set(self.tags).intersection(other.tags) == set(self.tags)) \
                                       or (self.tags is None and other.tags is None)

        namespace_condition = lambda other: (self.namespace is not None
                                             and other.namespace is not None
                                             and self.namespace == other.namespace) \
                                            or (self.namespace is None and other.namespace is None)

        return default_condition(other) and tags_condition(other) and namespace_condition(other)

    @staticmethod
    def from_string_format(string_format):
        registration_attributes = string_format.split(RegistrationInfo._ATTRIBUTE_SEPARATOR)
        registration_dict = {}
        for registration_attribute in registration_attributes:
            key, value = registration_attribute.split(RegistrationInfo._KEY_VALUE_SEPARATOR)
            if key == 'flag':
                value = ComponentFlag[value.upper()]
            if key == 'tags':
                value = ast.literal_eval(value)

            registration_dict[key] = value

        return RegistrationInfo(**registration_dict)


class ProjectRegistry(object, metaclass=MetaProjectRegistry):
    # (External) #

    # General
    PROJECT_DIR = ""

    # Logging
    NAME_LOG = 'daily_log.log'

    # Default Directories
    PATH_LOG = os.path.join(PROJECT_DIR, 'log')
    LOCAL_DATASETS_DIR = os.path.join(PROJECT_DIR, 'local_database')
    REGISTRATION_DIR = os.path.join(PROJECT_DIR, 'registrations')
    CALIBRATION_DIR = os.path.join(PROJECT_DIR, 'calibration')
    CALIBRATION_RESULTS_DIR = os.path.join(PROJECT_DIR, 'calibration_results')
    PREBUILT_FOLDS_DIR = os.path.join(PROJECT_DIR, 'prebuilt_folds')
    TESTS_DATA_DIR = os.path.join(PROJECT_DIR, 'tests_data')

    # Task
    TASK_DIR = os.path.join(PROJECT_DIR, 'tasks')

    # JSON files
    JSON_MODEL_DATA_CONFIGS_NAME = 'data_config.json'
    JSON_TASK_CONFIG_REGISTRATION_INFO_NAME = 'task_config_registration_info.json'
    JSON_CALIBRATION_RESULTS_NAME = 'calibration_results.json'

    # Pickle files
    PICKLE_TASK_CONFIG_NAME = 'task_config.pickle'

    # Constants

    CONSTANTS = {
        # Logging
        'logging_filename': NAME_LOG,

        # Default directories
        'logging_dir': PATH_LOG,
        'registration_dir': REGISTRATION_DIR,
        'local_database': LOCAL_DATASETS_DIR,
        'calibration_dir': CALIBRATION_DIR,
        'calibration_results_dir': CALIBRATION_RESULTS_DIR,
        'prebuilt_folds_dir': PREBUILT_FOLDS_DIR,
        'stored_data_dir': TESTS_DATA_DIR,

        # Evaluation routines
        'task_dir': TASK_DIR
    }

    COMPONENT_KEY = 'class'
    CONFIGURATION_KEY = 'instance'

    # (Internal) #

    _MODULES = set()

    # Components

    _REGISTRY = {}

    # Configuration APIs

    @staticmethod
    def get_current_dir():
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def _reload_directories():
        ProjectRegistry.PATH_LOG = os.path.join(ProjectRegistry.PROJECT_DIR, 'log')
        ProjectRegistry.LOCAL_DATASETS_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'local_database')
        ProjectRegistry.REGISTRATION_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'registrations')
        ProjectRegistry.CALIBRATION_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'calibration')
        ProjectRegistry.CALIBRATION_RESULTS_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'calibration_results')
        ProjectRegistry.PREBUILT_FOLDS_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'prebuilt_folds')
        ProjectRegistry.MONGO_DB_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'mongo_db')
        ProjectRegistry.TESTS_DATA_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'tests_data')

        ProjectRegistry.TASK_DIR = os.path.join(ProjectRegistry.PROJECT_DIR, 'tasks')

        ProjectRegistry.CONSTANTS = {
            # Logging
            'logging_filename': ProjectRegistry.NAME_LOG,

            # Default directories
            'logging_dir': ProjectRegistry.PATH_LOG,
            'registration_dir': ProjectRegistry.REGISTRATION_DIR,
            'local_database': ProjectRegistry.LOCAL_DATASETS_DIR,
            'calibration_dir': ProjectRegistry.CALIBRATION_DIR,
            'calibration_results_dir': ProjectRegistry.CALIBRATION_RESULTS_DIR,
            'prebuilt_folds_dir': ProjectRegistry.PREBUILT_FOLDS_DIR,
            'mongo_db_dir': ProjectRegistry.MONGO_DB_DIR,
            'stored_data_dir': ProjectRegistry.TESTS_DATA_DIR,

            # Evaluation routines
            'task_dir': ProjectRegistry.TASK_DIR
        }

    @staticmethod
    def set_project_dir(directory=None):
        if directory is not None:
            ProjectRegistry.PROJECT_DIR = directory
        else:
            ProjectRegistry.PROJECT_DIR = ProjectRegistry.get_current_dir()

        # Make sure we point to an existing directory
        assert os.path.isdir(ProjectRegistry.PROJECT_DIR)

        ProjectRegistry._reload_directories()

        from deasy_learning_generic.utility.log_utils import Logger
        Logger.get_logger(__name__).info('Setting project folder to: {}'.format(ProjectRegistry.PROJECT_DIR))

    @staticmethod
    def update_constants_from_file(config_path, show_values=False):
        config_data = load_json(config_path)
        ProjectRegistry.update_constants_from_dict(config_data=config_data,
                                                   show_values=show_values)

    @staticmethod
    def update_constants_from_dict(config_data, show_values=False):
        from deasy_learning_generic.utility.log_utils import Logger
        logger = Logger.get_logger(__name__)

        for key, value in config_data.items():
            if key in ProjectRegistry.CONSTANTS:
                ProjectRegistry.CONSTANTS[key] = value
            else:
                logger.info('Adding constant key {0}, since it was not found'.format(key))
                ProjectRegistry.CONSTANTS[key] = value

        if show_values:
            ProjectRegistry.show_constants()

    @staticmethod
    def _build(class_type, constructor=None, args=None):
        args = args if args is not None else {}

        if constructor is not None:
            return constructor(**args)
        else:
            return class_type(**args)

    @staticmethod
    def load_custom_module(module_name):
        importlib.import_module(module_name)
        ProjectRegistry._MODULES.add(module_name)

    @staticmethod
    def get_loaded_modules():
        return ProjectRegistry._MODULES

    # Registration APIs

    @staticmethod
    def _lookup_element(internal_key, flag, framework, tags=None, namespace=None, exact_match=True):
        search_info = RegistrationInfo(flag=flag,
                                       internal_key=internal_key,
                                       framework=framework,
                                       tags=tags,
                                       namespace=namespace)
        if exact_match:
            assert search_info in ProjectRegistry._REGISTRY, \
                f"Expected to find a registered element with registration info: {search_info}. Search failed."
            return ProjectRegistry._REGISTRY[search_info]
        else:
            return [info for info in ProjectRegistry._REGISTRY if info.partial_match(search_info)]

    @staticmethod
    def _lookup_element_from_info(search_info, exact_match=True):
        if exact_match:
            assert search_info in ProjectRegistry._REGISTRY, \
                f"Expected to find a registered element with registration info: {search_info}. Search failed."
            return ProjectRegistry._REGISTRY[search_info]
        else:
            return [info for info in ProjectRegistry._REGISTRY if info.partial_match(search_info)]

    # Component

    @staticmethod
    def register_component(class_type, framework, flag, tags=None, namespace=None, constructor=None):
        assert isinstance(flag, ComponentFlag)

        registration_info = RegistrationInfo(flag=flag, tags=tags,
                                             framework=framework,
                                             namespace=namespace,
                                             internal_key=ProjectRegistry.COMPONENT_KEY)
        registered_component = RegisteredComponent(class_type=class_type, constructor=constructor)

        # Check if already registered
        if registration_info in ProjectRegistry._REGISTRY:
            raise RuntimeError('A component has already been registered with the same registration info!'
                               f'Got: {registration_info}')

        # Store component in registry
        ProjectRegistry._REGISTRY.setdefault(registration_info, registered_component)

    @staticmethod
    def retrieve_component(flag, args=None, framework=None, tags=None, namespace=None):
        assert isinstance(flag, ComponentFlag)
        retrieved_component = ProjectRegistry._lookup_element(internal_key=ProjectRegistry.COMPONENT_KEY,
                                                              flag=flag,
                                                              framework=framework,
                                                              tags=tags,
                                                              namespace=namespace,
                                                              exact_match=True)

        args = args if args is not None else {}
        return ProjectRegistry._build(class_type=retrieved_component.class_type,
                                      constructor=retrieved_component.constructor,
                                      args=args)

    @staticmethod
    def retrieve_component_from_info(registration_info, args=None, build=True):
        retrieved_component = ProjectRegistry._lookup_element_from_info(search_info=registration_info,
                                                                        exact_match=True)
        args = args if args is not None else {}

        if build:
            return ProjectRegistry._build(class_type=retrieved_component.class_type,
                                          constructor=retrieved_component.constructor,
                                          args=args)
        else:
            return retrieved_component

    # Configuration

    @staticmethod
    def register_configuration(configuration, framework, tags=None, namespace=None):
        flag = configuration.get_configuration_flag()
        assert isinstance(flag, ComponentFlag)

        registration_info = RegistrationInfo(flag=flag, tags=tags,
                                             framework=framework,
                                             namespace=namespace,
                                             internal_key=ProjectRegistry.CONFIGURATION_KEY)
        registered_configuration = RegisteredConfiguration(config=configuration)

        # Check if already registered
        if registration_info in ProjectRegistry._REGISTRY:
            raise RuntimeError('A configuration has already been registered with the same registration info!'
                               f'Got: {registration_info}')

        # Store configuration in registry
        ProjectRegistry._REGISTRY.setdefault(registration_info, registered_configuration)

    @staticmethod
    def retrieve_configurations(flag, framework=None, tags=None, namespace=None, exact_match=True):
        assert isinstance(flag, ComponentFlag)
        retrieved_configurations = ProjectRegistry._lookup_element(internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                                   flag=flag,
                                                                   framework=framework,
                                                                   tags=tags,
                                                                   namespace=namespace,
                                                                   exact_match=exact_match)
        if exact_match:
            return retrieved_configurations.config
        else:
            return [config.config for config in retrieved_configurations]

    @staticmethod
    def retrieve_configurations_from_info(registration_info, exact_match=True):
        retrieved_configurations = ProjectRegistry._lookup_element_from_info(search_info=registration_info,
                                                                             exact_match=exact_match)
        if exact_match:
            return retrieved_configurations.config
        else:
            return [config.config for config in retrieved_configurations]

    # Utility

    @staticmethod
    def show_constants():
        from deasy_learning_generic.utility.log_utils import Logger
        logger = Logger.get_logger(__name__)

        logger.info('Showing project constants...')
        for key, value in ProjectRegistry.CONSTANTS.items():
            logger.info('{0} --> {1}'.format(key, value))

    @staticmethod
    def get_registered_elements(flag_or_flag_filters=None,
                                framework_or_framework_filters=None,
                                tags_or_tags_filters=None,
                                namespace_or_namespace_filters=None,
                                internal_key_or_internal_key_filters=None):
        filters = []

        flag_or_flag_filters = parse_and_create_filter(data=flag_or_flag_filters, attribute_name='flag')
        framework_or_framework_filters = parse_and_create_filter(data=framework_or_framework_filters,
                                                                 attribute_name='framework')
        tags_or_tags_filters = parse_and_create_filter(data=tags_or_tags_filters, attribute_name='tags')
        namespace_or_namespace_filters = parse_and_create_filter(data=namespace_or_namespace_filters,
                                                                 attribute_name='namespace')
        internal_key_or_internal_key_filters = parse_and_create_filter(data=internal_key_or_internal_key_filters,
                                                                       attribute_name='internal_key')

        filters += flag_or_flag_filters + framework_or_framework_filters + \
                   tags_or_tags_filters + namespace_or_namespace_filters + \
                   internal_key_or_internal_key_filters

        # We accept everything in case of no filter options
        if not len(filters):
            filters.append(PassFilter())

        filtered_info = list(filter(lambda element: all(map(lambda f: f(element), filters)), ProjectRegistry._REGISTRY))
        return [(info, ProjectRegistry._REGISTRY[info]) for info in filtered_info]
