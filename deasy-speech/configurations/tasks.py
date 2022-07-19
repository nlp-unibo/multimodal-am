from deasy_learning_generic.configuration import TaskConfiguration
from deasy_learning_generic.registry import ProjectRegistry, RegistrationInfo, ComponentFlag


class ArgAAAITaskConfiguration(TaskConfiguration):

    @classmethod
    def get_default(cls):
        task_config = cls(
            component_registration_info=RegistrationInfo(
                flag=ComponentFlag.TASK,
                framework='generic',
                namespace='default',
                internal_key=ProjectRegistry.COMPONENT_KEY
            ),
            task_name='arg_aaai')

        # Evaluation
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       tags=['default', 'training'],
                                                       namespace='arg_aaai',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.EVALUATION))

        # Routine
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       namespace='arg_aaai',
                                                       tags=['default', 'cv_test'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.ROUTINE))

        # Framework
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['first_gpu'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.FRAMEWORK_HELPER))

        # Callbacks
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['default', 'early_stopping'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.CALLBACK),
                                      allow_multiple=True,
                                      key='early_stopping')
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['default', 'training_logger'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.CALLBACK),
                                      key='training_logger',
                                      allow_multiple=True)

        # Data Loader
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       tags=['default'],
                                                       namespace='arg_aaai',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.DATA_LOADER))

        # Model
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       tags=['text_only', 'lstm'],
                                                       namespace='arg_aaai',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.MODEL))

        task_config.evaluate()

        return task_config

    def get_debug_version(self):
        # Evaluation
        self.add_configuration(RegistrationInfo(framework='generic',
                                                flag=ComponentFlag.EVALUATION,
                                                tags=['debug'],
                                                namespace='default',
                                                internal_key=ProjectRegistry.CONFIGURATION_KEY),
                               force_update=True)

        # Framework
        self.add_configuration(RegistrationInfo(framework='tf',
                                                tags=['debug', 'eager_execution'],
                                                namespace='default',
                                                internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                flag=ComponentFlag.FRAMEWORK_HELPER),
                               force_update=True)


def register_arg_aaai_task_configurations():
    default_config = ArgAAAITaskConfiguration.get_default()

    # models
    models = ProjectRegistry.get_registered_elements(flag_or_flag_filters=ComponentFlag.MODEL,
                                                     namespace_or_namespace_filters='arg_aaai',
                                                     internal_key_or_internal_key_filters=ProjectRegistry.CONFIGURATION_KEY)

    for model_config_registration_info, model_configuration in models:
        current_task_config = default_config.get_delta_copy()
        current_task_config.add_configuration(model_config_registration_info, force_update=True)

        if 'bert' in model_config_registration_info.tags:
            current_task_config.add_configuration(RegistrationInfo(framework='generic',
                                                                   tags=['bert', 'training'],
                                                                   namespace='arg_aaai',
                                                                   internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                                   flag=ComponentFlag.EVALUATION),
                                                  force_update=True)

        ProjectRegistry.register_configuration(configuration=current_task_config,
                                               namespace='arg_aaai',
                                               tags=model_config_registration_info.tags,
                                               framework='tf')


class MArgTaskConfiguration(TaskConfiguration):

    @classmethod
    def get_default(cls):
        task_config = cls(
            component_registration_info=RegistrationInfo(
                flag=ComponentFlag.TASK,
                framework='generic',
                namespace='default',
                internal_key=ProjectRegistry.COMPONENT_KEY
            ),
            task_name='m-arg')

        # Evaluation
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       tags=['default', 'training'],
                                                       namespace='m-arg',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.EVALUATION))

        # Routine
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       namespace='m-arg',
                                                       tags=['default', 'cv_test'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.ROUTINE))

        # Framework
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['first_gpu'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.FRAMEWORK_HELPER))

        # Callbacks
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='m_arg',
                                                       tags=['f1', 'early_stopping'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.CALLBACK),
                                      key='early_stopping',
                                      allow_multiple=True)
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['default', 'training_logger'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.CALLBACK),
                                      key='training_logger',
                                      allow_multiple=True)

        # Data Loader
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       tags=['default'],
                                                       namespace='m-arg',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.DATA_LOADER))

        # Model
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       tags=['annotation_confidence=0.0',
                                                             'text_only', 'lstm'],
                                                       namespace='m-arg',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.MODEL))

        task_config.evaluate()

        return task_config

    def get_debug_version(self):
        # Evaluation
        self.add_configuration(RegistrationInfo(framework='generic',
                                                flag=ComponentFlag.EVALUATION,
                                                tags=['debug'],
                                                namespace='default',
                                                internal_key=ProjectRegistry.CONFIGURATION_KEY),
                               force_update=True)

        # Framework
        self.add_configuration(RegistrationInfo(framework='tf',
                                                tags=['debug', 'eager_execution'],
                                                namespace='default',
                                                internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                flag=ComponentFlag.FRAMEWORK_HELPER),
                               force_update=True)


def register_m_arg_task_configurations():
    default_config = MArgTaskConfiguration.get_default()

    # models
    models = ProjectRegistry.get_registered_elements(flag_or_flag_filters=ComponentFlag.MODEL,
                                                     namespace_or_namespace_filters='m-arg',
                                                     internal_key_or_internal_key_filters=ProjectRegistry.CONFIGURATION_KEY)

    for model_config_registration_info, model_configuration in models:
        current_task_config = default_config.get_delta_copy()
        current_task_config.add_configuration(model_config_registration_info, force_update=True)

        if 'bert' in model_config_registration_info.tags:
            current_task_config.add_configuration(RegistrationInfo(framework='generic',
                                                                   tags=['bert', 'training'],
                                                                   namespace='m-arg',
                                                                   internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                                   flag=ComponentFlag.EVALUATION),
                                                  force_update=True)

        ProjectRegistry.register_configuration(configuration=current_task_config,
                                               namespace='m-arg',
                                               tags=model_config_registration_info.tags,
                                               framework='tf')


class UsElecTaskConfiguration(TaskConfiguration):

    @classmethod
    def get_default(cls):
        task_config = cls(
            component_registration_info=RegistrationInfo(
                flag=ComponentFlag.TASK,
                framework='generic',
                namespace='default',
                internal_key=ProjectRegistry.COMPONENT_KEY
            ),
            task_name='us_elec')

        # Evaluation
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       tags=['default', 'training'],
                                                       namespace='us_elec',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.EVALUATION))

        # Routine
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       namespace='us_elec',
                                                       tags=['default', 'train_and_test'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.ROUTINE))

        # Framework
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['first_gpu'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.FRAMEWORK_HELPER))

        # Callbacks
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['default', 'early_stopping'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.CALLBACK),
                                      allow_multiple=True,
                                      key='early_stopping')
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       namespace='default',
                                                       tags=['default', 'training_logger'],
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.CALLBACK),
                                      key='training_logger',
                                      allow_multiple=True)

        # Data Loader
        task_config.add_configuration(RegistrationInfo(framework='generic',
                                                       tags=['default'],
                                                       namespace='us_elec',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.DATA_LOADER))

        # Model
        task_config.add_configuration(RegistrationInfo(framework='tf',
                                                       tags=['text_only', 'lstm', 'task_type=asd'],
                                                       namespace='us_elec',
                                                       internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                       flag=ComponentFlag.MODEL))

        task_config.evaluate()

        return task_config

    def get_debug_version(self):
        # Evaluation
        self.add_configuration(RegistrationInfo(framework='generic',
                                                flag=ComponentFlag.EVALUATION,
                                                tags=['debug', 'training'],
                                                namespace='us_elec',
                                                internal_key=ProjectRegistry.CONFIGURATION_KEY),
                               force_update=True)

        # Framework
        # self.add_configuration(RegistrationInfo(framework='tf',
        #                                         tags=['debug', 'eager_execution'],
        #                                         namespace='default',
        #                                         internal_key=ProjectRegistry.CONFIGURATION_KEY,
        #                                         flag=ComponentFlag.FRAMEWORK_HELPER),
        #                        force_update=True)


def register_us_elec_task_configurations():
    default_config = UsElecTaskConfiguration.get_default()

    # models
    models = ProjectRegistry.get_registered_elements(flag_or_flag_filters=ComponentFlag.MODEL,
                                                     namespace_or_namespace_filters='us_elec',
                                                     internal_key_or_internal_key_filters=ProjectRegistry.CONFIGURATION_KEY)

    for model_config_registration_info, model_configuration in models:
        current_task_config = default_config.get_delta_copy()
        current_task_config.add_configuration(model_config_registration_info, force_update=True)

        if 'bert' in model_config_registration_info.tags:
            current_task_config.add_configuration(RegistrationInfo(framework='generic',
                                                                   tags=['bert', 'training'],
                                                                   namespace='us_elec',
                                                                   internal_key=ProjectRegistry.CONFIGURATION_KEY,
                                                                   flag=ComponentFlag.EVALUATION),
                                                  force_update=True)

        ProjectRegistry.register_configuration(configuration=current_task_config,
                                               namespace='us_elec',
                                               tags=model_config_registration_info.tags,
                                               framework='tf')


def register_task_configurations():
    register_arg_aaai_task_configurations()
    register_m_arg_task_configurations()
    register_us_elec_task_configurations()
