from deasy_learning_generic.configuration import DataLoaderConfiguration
from deasy_learning_generic.implementations.metrics import SklearnMetric
from deasy_learning_generic.registry import RegistrationInfo, ComponentFlag, ProjectRegistry


class ArgAAAIConfiguration(DataLoaderConfiguration):

    def __init__(self, mode='all', mfccs=25, use_audio_features=True, **kwargs):
        super(ArgAAAIConfiguration, self).__init__(**kwargs)
        self.mfccs = mfccs
        self.mode = mode
        self.use_audio_features = use_audio_features

    @classmethod
    def get_default(cls):
        metrics = [
            SklearnMetric(name='binary_F1',
                          function_name='f1_score',
                          metric_arguments={'average': 'binary',
                                            'pos_label': 1}),
            SklearnMetric(name='accuracy', function_name='accuracy_score', metric_arguments={}),
            SklearnMetric(name='all_F1', function_name='f1_score',
                          metric_arguments={'average': None}),
        ]

        return cls(label_metrics_map=None, metrics=metrics, mfccs=25, mode='all',
                   component_registration_info=RegistrationInfo(flag=ComponentFlag.DATA_LOADER,
                                                                framework='generic',
                                                                namespace='arg_aaai',
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))


def register_arg_aaai_data_loader_configurations():
    ProjectRegistry.register_configuration(configuration=ArgAAAIConfiguration.get_default(),
                                           namespace='arg_aaai',
                                           tags=['default'],
                                           framework='generic')


class MArgConfiguration(DataLoaderConfiguration):

    def __init__(self, annotation_confidence=0., mfccs=25, use_audio_features=True, **kwargs):
        super(MArgConfiguration, self).__init__(**kwargs)
        self.annotation_confidence = annotation_confidence
        self.mfccs = mfccs
        self.use_audio_features = use_audio_features

    @classmethod
    def get_default(cls):
        metrics = [
            # We only care about positive classes
            SklearnMetric(name='macro_F1', function_name='f1_score',
                          metric_arguments={'average': 'macro',
                                            'labels': [1, 2]}),
            SklearnMetric(name='accuracy', function_name='accuracy_score', metric_arguments={}),
            SklearnMetric(name='all_F1', function_name='f1_score',
                          metric_arguments={'average': None}),
        ]

        return cls(label_metrics_map=None, metrics=metrics, mfccs=25, annotation_confidence=0.,
                   component_registration_info=RegistrationInfo(flag=ComponentFlag.DATA_LOADER,
                                                                framework='generic',
                                                                namespace='m-arg',
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))


def register_m_arg_data_loader_configurations():
    ProjectRegistry.register_configuration(configuration=MArgConfiguration.get_default(),
                                           namespace='m-arg',
                                           tags=['default'],
                                           framework='generic')


class UsElecConfiguration(DataLoaderConfiguration):

    def __init__(self, task_type='asd', data_mode='text_only', mfccs=25, use_audio_features=True, **kwargs):
        super(UsElecConfiguration, self).__init__(**kwargs)
        self.task_type = task_type
        self.data_mode = data_mode
        self.mfccs = mfccs
        self.use_audio_features = use_audio_features

    @classmethod
    def get_default(cls):
        metrics = [
            SklearnMetric(name='macro_F1',
                          function_name='f1_score',
                          metric_arguments={'average': 'macro'}),
            SklearnMetric(name='accuracy', function_name='accuracy_score', metric_arguments={}),
            SklearnMetric(name='all_F1', function_name='f1_score',
                          metric_arguments={'average': None}),
        ]

        return cls(label_metrics_map=None, metrics=metrics, mfccs=25,
                   component_registration_info=RegistrationInfo(flag=ComponentFlag.DATA_LOADER,
                                                                framework='generic',
                                                                namespace='us_elec',
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))


def register_us_elec_data_loader_configurations():
    ProjectRegistry.register_configuration(configuration=UsElecConfiguration.get_default(),
                                           namespace='us_elec',
                                           tags=['default'],
                                           framework='generic')


def register_data_loader_configurations():
    register_arg_aaai_data_loader_configurations()
    register_m_arg_data_loader_configurations()
    register_us_elec_data_loader_configurations()
