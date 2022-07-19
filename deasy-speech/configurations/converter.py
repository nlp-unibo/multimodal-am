from deasy_learning_generic.registry import RegistrationInfo, ProjectRegistry, ComponentFlag
from deasy_learning_tf.implementations.nlp.configuration import TFTextConverterConfiguration


class ArgAAAIConverterConfiguration(TFTextConverterConfiguration):

    def __init__(self, max_frame_limit=None, data_mode='text_only', **kwargs):
        super(ArgAAAIConverterConfiguration, self).__init__(**kwargs)
        self.max_frame_limit = max_frame_limit
        self.data_mode = data_mode

    @classmethod
    def get_default(cls):
        return cls(children=[RegistrationInfo(tags=['default', 'text'],
                                              framework='tf',
                                              namespace='default',
                                              flag=ComponentFlag.TOKENIZER,
                                              internal_key=ProjectRegistry.CONFIGURATION_KEY)],
                   checkpoint=None,
                   feature_registration_info=RegistrationInfo(framework='tf',
                                                              namespace='arg_aaai',
                                                              flag=ComponentFlag.FEATURE,
                                                              internal_key=ProjectRegistry.COMPONENT_KEY),
                   component_registration_info=RegistrationInfo(framework='tf',
                                                                namespace='arg_aaai',
                                                                flag=ComponentFlag.CONVERTER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(ArgAAAIConverterConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['max_frame_limit'] = self.max_frame_limit
        return parameters


class ArgAAAIBertConverterConfiguration(ArgAAAIConverterConfiguration):

    @classmethod
    def get_default(cls):
        return cls(children=[RegistrationInfo(tags=['bert', 'default'],
                                              framework='generic',
                                              namespace='transformers',
                                              flag=ComponentFlag.TOKENIZER,
                                              internal_key=ProjectRegistry.CONFIGURATION_KEY)],
                   checkpoint=None,
                   feature_registration_info=RegistrationInfo(framework='tf',
                                                              namespace='arg_aaai',
                                                              tags=['bert'],
                                                              flag=ComponentFlag.FEATURE,
                                                              internal_key=ProjectRegistry.COMPONENT_KEY),
                   component_registration_info=RegistrationInfo(framework='tf',
                                                                namespace='arg_aaai',
                                                                tags=['bert'],
                                                                flag=ComponentFlag.CONVERTER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(ArgAAAIBertConverterConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['max_frame_limit'] = self.max_frame_limit
        return parameters


def register_arg_aaai_converter_configurations():
    default_config = ArgAAAIConverterConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['default'])

    # Audio only doesn't require tokenizer
    audio_only_config = default_config.get_delta_copy(children=[])
    ProjectRegistry.register_configuration(configuration=audio_only_config,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['no_tokenizer'])

    bert_default_config = ArgAAAIBertConverterConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=bert_default_config,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['bert'])

    # Audio only doesn't require tokenizer
    bert_audio_only_config = bert_default_config.get_delta_copy(children=[])
    ProjectRegistry.register_configuration(configuration=bert_audio_only_config,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['no_tokenizer', 'bert'])


class MArgConverterConfiguration(TFTextConverterConfiguration):

    def __init__(self, max_frame_limit=None, data_mode='text_only', **kwargs):
        super(MArgConverterConfiguration, self).__init__(**kwargs)
        self.max_frame_limit = max_frame_limit
        self.data_mode = data_mode

    @classmethod
    def get_default(cls):
        return cls(children=[RegistrationInfo(tags=['default', 'text'],
                                              framework='tf',
                                              namespace='default',
                                              flag=ComponentFlag.TOKENIZER,
                                              internal_key=ProjectRegistry.CONFIGURATION_KEY)],
                   checkpoint=None,
                   feature_registration_info=RegistrationInfo(framework='tf',
                                                              namespace='m-arg',
                                                              flag=ComponentFlag.FEATURE,
                                                              internal_key=ProjectRegistry.COMPONENT_KEY),
                   component_registration_info=RegistrationInfo(framework='tf',
                                                                namespace='m-arg',
                                                                flag=ComponentFlag.CONVERTER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(MArgConverterConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['max_frame_limit'] = self.max_frame_limit
        return parameters


class MArgBertConverterConfiguration(MArgConverterConfiguration):

    @classmethod
    def get_default(cls):
        return cls(children=[RegistrationInfo(tags=['bert', 'default'],
                                              framework='generic',
                                              namespace='transformers',
                                              flag=ComponentFlag.TOKENIZER,
                                              internal_key=ProjectRegistry.CONFIGURATION_KEY)],
                   checkpoint=None,
                   feature_registration_info=RegistrationInfo(framework='tf',
                                                              tags=['bert'],
                                                              namespace='m-arg',
                                                              flag=ComponentFlag.FEATURE,
                                                              internal_key=ProjectRegistry.COMPONENT_KEY),
                   component_registration_info=RegistrationInfo(framework='tf',
                                                                tags=['bert'],
                                                                namespace='m-arg',
                                                                flag=ComponentFlag.CONVERTER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(MArgBertConverterConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['max_frame_limit'] = self.max_frame_limit
        return parameters


def register_m_arg_converter_configurations():
    default_config = MArgConverterConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           framework='tf',
                                           namespace='m-arg',
                                           tags=['default'])

    # Audio only doesn't require tokenizer
    audio_only_config = default_config.get_delta_copy(children=[])
    ProjectRegistry.register_configuration(configuration=audio_only_config,
                                           framework='tf',
                                           namespace='m-arg',
                                           tags=['no_tokenizer'])

    bert_default_config = MArgBertConverterConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=bert_default_config,
                                           framework='tf',
                                           namespace='m-arg',
                                           tags=['bert'])

    # Audio only doesn't require tokenizer
    bert_audio_only_config = bert_default_config.get_delta_copy(children=[])
    ProjectRegistry.register_configuration(configuration=bert_audio_only_config,
                                           framework='tf',
                                           namespace='m-arg',
                                           tags=['no_tokenizer', 'bert'])


class UsElecConverterConfiguration(TFTextConverterConfiguration):

    def __init__(self, max_frame_limit=None, data_mode='text_only', **kwargs):
        super(UsElecConverterConfiguration, self).__init__(**kwargs)
        self.max_frame_limit = max_frame_limit
        self.data_mode = data_mode

    @classmethod
    def get_default(cls):
        return cls(children=[RegistrationInfo(tags=['default', 'text'],
                                              framework='tf',
                                              namespace='default',
                                              flag=ComponentFlag.TOKENIZER,
                                              internal_key=ProjectRegistry.CONFIGURATION_KEY)],
                   checkpoint=None,
                   feature_registration_info=RegistrationInfo(framework='tf',
                                                              namespace='us_elec',
                                                              flag=ComponentFlag.FEATURE,
                                                              internal_key=ProjectRegistry.COMPONENT_KEY),
                   component_registration_info=RegistrationInfo(framework='tf',
                                                                namespace='us_elec',
                                                                flag=ComponentFlag.CONVERTER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(UsElecConverterConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['max_frame_limit'] = self.max_frame_limit
        return parameters


class UsElecBertConverterConfiguration(UsElecConverterConfiguration):

    @classmethod
    def get_default(cls):
        return cls(children=[RegistrationInfo(tags=['bert', 'default'],
                                              framework='generic',
                                              namespace='transformers',
                                              flag=ComponentFlag.TOKENIZER,
                                              internal_key=ProjectRegistry.CONFIGURATION_KEY)],
                   checkpoint=None,
                   feature_registration_info=RegistrationInfo(framework='tf',
                                                              namespace='us_elec',
                                                              tags=['bert'],
                                                              flag=ComponentFlag.FEATURE,
                                                              internal_key=ProjectRegistry.COMPONENT_KEY),
                   component_registration_info=RegistrationInfo(framework='tf',
                                                                namespace='us_elec',
                                                                tags=['bert'],
                                                                flag=ComponentFlag.CONVERTER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(UsElecBertConverterConfiguration, self).get_serialization_parameters()
        parameters['data_mode'] = self.data_mode
        parameters['max_frame_limit'] = self.max_frame_limit
        return parameters


def register_us_elec_converter_configurations():
    default_config = UsElecConverterConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['default'])

    # Audio only doesn't require tokenizer
    audio_only_config = default_config.get_delta_copy(children=[])
    ProjectRegistry.register_configuration(configuration=audio_only_config,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['no_tokenizer'])

    bert_default_config = UsElecBertConverterConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=bert_default_config,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['bert'])

    # Audio only doesn't require tokenizer
    bert_audio_only_config = bert_default_config.get_delta_copy(children=[])
    ProjectRegistry.register_configuration(configuration=bert_audio_only_config,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['no_tokenizer', 'bert'])


def register_converter_configurations():
    register_arg_aaai_converter_configurations()
    register_m_arg_converter_configurations()
    register_us_elec_converter_configurations()
