import os

from deasy_learning_generic.configuration import CVTestRoutineConfiguration, RoutineConfiguration
from deasy_learning_generic.registry import RegistrationInfo, ComponentFlag, ProjectRegistry


class ArgAAAICvTestConfiguration(CVTestRoutineConfiguration):

    def __init__(self, mode='all', **kwargs):
        super(ArgAAAICvTestConfiguration, self).__init__(**kwargs)
        self.mode = mode

    @classmethod
    def get_default(cls):
        return cls(repetitions=3,
                   split_key='idx',
                   cv_type='kfold',
                   n_splits=5,
                   compute_test_info=True,
                   component_registration_info=RegistrationInfo(
                       flag=ComponentFlag.ROUTINE,
                       framework='generic',
                       tags=['cv_test'],
                       namespace='arg_aaai',
                       internal_key=ProjectRegistry.COMPONENT_KEY
                   ),
                   seeds=[15371, 15372, 15373],
                   validation_percentage=0.20)


def register_arg_aaai_routine_configurations():
    default_config = ArgAAAICvTestConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           namespace='arg_aaai',
                                           tags=['default', 'cv_test'],
                                           framework='generic')

    # 1 repetition
    one_rep_config = default_config.get_delta_copy(repetitions=1, seeds=[15371])
    ProjectRegistry.register_configuration(configuration=one_rep_config,
                                           namespace='arg_aaai',
                                           tags=['one_rep', 'cv_test'],
                                           framework='generic')


class MArgCvTestConfiguration(CVTestRoutineConfiguration):

    def __init__(self, annotation_confidence=0.00, **kwargs):
        super(MArgCvTestConfiguration, self).__init__(**kwargs)
        self.annotation_confidence = annotation_confidence

    @classmethod
    def get_default(cls):
        return cls(repetitions=3,
                   split_key='index',
                   cv_type='kfold',
                   folds_path=os.path.join(ProjectRegistry['prebuilt_folds_dir'],
                                           'm_arg_folds_0.00.json'),
                   n_splits=5,
                   compute_test_info=True,
                   component_registration_info=RegistrationInfo(
                       flag=ComponentFlag.ROUTINE,
                       framework='generic',
                       tags=['cv_test'],
                       namespace='m-arg',
                       internal_key=ProjectRegistry.COMPONENT_KEY
                   ),
                   seeds=[15371, 15372, 15373],
                   validation_percentage=None)


def register_m_arg_routine_configurations():
    default_config = MArgCvTestConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           namespace='m-arg',
                                           tags=['default', 'cv_test'],
                                           framework='generic')

    # Confidence = 0.85
    confidence_config = default_config.get_delta_copy(annotation_confidence=0.85)
    ProjectRegistry.register_configuration(configuration=confidence_config,
                                           namespace='m-arg',
                                           tags=['confidence=0.85', 'cv_test'],
                                           framework='generic')


class UsElecRoutineConfiguration(RoutineConfiguration):

    @classmethod
    def get_default(cls):
        return cls(repetitions=3,
                   seeds=[15371, 15372, 15373],
                   validation_percentage=None,
                   component_registration_info=RegistrationInfo(
                       flag=ComponentFlag.ROUTINE,
                       framework='generic',
                       tags=['train_and_test'],
                       namespace='default',
                       internal_key=ProjectRegistry.COMPONENT_KEY
                   ),
                   compute_test_info=True)


def register_us_elec_routine_configurations():
    default_config = UsElecRoutineConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           namespace='us_elec',
                                           tags=['default', 'train_and_test'],
                                           framework='generic')

    # 1 repetition
    one_rep_config = default_config.get_delta_copy(repetitions=1, seeds=[15371])
    ProjectRegistry.register_configuration(configuration=one_rep_config,
                                           namespace='us_elec',
                                           tags=['one_rep', 'train_and_test'],
                                           framework='generic')


def register_routine_configurations():
    register_arg_aaai_routine_configurations()
    register_m_arg_routine_configurations()
    register_us_elec_routine_configurations()
