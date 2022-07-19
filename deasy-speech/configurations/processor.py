from deasy_learning_generic.configuration import ProcessorConfiguration
from deasy_learning_generic.registry import RegistrationInfo, ComponentFlag, ProjectRegistry


class ArgAAAIProcessorConfiguration(ProcessorConfiguration):

    def __init__(self, filter_names=None, retrieve_label=True, disable_filtering=False,
                 data_mode='text_only', pooling_sizes=None, normalize_mfccs=False,
                 use_audio_features=True, audio_model_sampling_rate=None, audio_model_name=None, **kwargs):
        super(ArgAAAIProcessorConfiguration, self).__init__(**kwargs)
        self.filter_names = filter_names
        self.disable_filtering = disable_filtering
        self.retrieve_label = retrieve_label
        self.data_mode = data_mode
        self.pooling_sizes = pooling_sizes
        self.normalize_mfccs = normalize_mfccs
        self.use_audio_features = use_audio_features
        self.audio_model_sampling_rate = audio_model_sampling_rate
        self.audio_model_name = audio_model_name

    @classmethod
    def get_default(cls):
        return cls(component_registration_info=RegistrationInfo(framework='generic',
                                                                namespace='arg_aaai',
                                                                flag=ComponentFlag.PROCESSOR,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(ArgAAAIProcessorConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['pooling_sizes'] = self.pooling_sizes
        parameters['normalize_mfccs'] = self.normalize_mfccs
        parameters['use_audio_features'] = self.use_audio_features
        parameters['audio_model_sampling_rate'] = self.audio_model_sampling_rate
        parameters['audio_model_name'] = self.audio_model_name

        return parameters


def register_arg_aaai_processor_configurations():
    # Default - no further preprocessing
    ProjectRegistry.register_configuration(configuration=ArgAAAIProcessorConfiguration.get_default(),
                                           framework='generic',
                                           namespace='arg_aaai',
                                           tags=['default'])


class MArgProcessorConfiguration(ProcessorConfiguration):

    def __init__(self, filter_names=None, retrieve_label=True, disable_filtering=False,
                 data_mode='text_only', pooling_sizes=None, normalize_mfccs=False,
                 use_audio_features=True, audio_model_sampling_rate=None, audio_model_name=None, **kwargs):
        super(MArgProcessorConfiguration, self).__init__(**kwargs)
        self.filter_names = filter_names
        self.disable_filtering = disable_filtering
        self.retrieve_label = retrieve_label
        self.data_mode = data_mode
        self.pooling_sizes = pooling_sizes
        self.normalize_mfccs = normalize_mfccs
        self.use_audio_features = use_audio_features
        self.audio_model_sampling_rate = audio_model_sampling_rate
        self.audio_model_name = audio_model_name

    @classmethod
    def get_default(cls):
        return cls(component_registration_info=RegistrationInfo(framework='generic',
                                                                namespace='m-arg',
                                                                flag=ComponentFlag.PROCESSOR,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(MArgProcessorConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['pooling_sizes'] = self.pooling_sizes
        parameters['normalize_mfccs'] = self.normalize_mfccs
        parameters['use_audio_features'] = self.use_audio_features
        parameters['audio_model_sampling_rate'] = self.audio_model_sampling_rate
        parameters['audio_model_name'] = self.audio_model_name

        return parameters


def register_m_arg_processor_configurations():
    # Default - no further preprocessing
    ProjectRegistry.register_configuration(configuration=MArgProcessorConfiguration.get_default(),
                                           framework='generic',
                                           namespace='m-arg',
                                           tags=['default'])


class UsElecProcessorConfiguration(ProcessorConfiguration):

    def __init__(self, filter_names=None, retrieve_label=True, disable_filtering=False,
                 data_mode='text_only', pooling_sizes=None, normalize_mfccs=False,
                 use_audio_features=True, audio_model_sampling_rate=None, audio_model_name=None, **kwargs):
        super(UsElecProcessorConfiguration, self).__init__(**kwargs)
        self.filter_names = filter_names
        self.disable_filtering = disable_filtering
        self.retrieve_label = retrieve_label
        self.data_mode = data_mode
        self.pooling_sizes = pooling_sizes
        self.normalize_mfccs = normalize_mfccs
        self.use_audio_features = use_audio_features
        self.audio_model_sampling_rate = audio_model_sampling_rate
        self.audio_model_name = audio_model_name

    @classmethod
    def get_default(cls):
        return cls(component_registration_info=RegistrationInfo(framework='generic',
                                                                namespace='us_elec',
                                                                flag=ComponentFlag.PROCESSOR,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(UsElecProcessorConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['pooling_sizes'] = self.pooling_sizes
        parameters['normalize_mfccs'] = self.normalize_mfccs
        parameters['use_audio_features'] = self.use_audio_features
        parameters['audio_model_sampling_rate'] = self.audio_model_sampling_rate
        parameters['audio_model_name'] = self.audio_model_name

        return parameters


def register_us_elec_processor_configurations():
    # Default - no further preprocessing
    ProjectRegistry.register_configuration(configuration=UsElecProcessorConfiguration.get_default(),
                                           framework='generic',
                                           namespace='us_elec',
                                           tags=['default'])


def register_processor_configurations():
    register_arg_aaai_processor_configurations()
    register_m_arg_processor_configurations()
    register_us_elec_processor_configurations()
