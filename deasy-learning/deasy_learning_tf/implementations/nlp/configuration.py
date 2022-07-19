from deasy_learning_tf.configuration import TFConverterConfiguration
from deasy_learning_generic.nlp.configuration import TokenizerConfiguration
from deasy_learning_generic.registry import ComponentFlag, RegistrationInfo, ProjectRegistry


class KerasTokenizerConfiguration(TokenizerConfiguration):

    def __init__(self, tokenizer_args=None, **kwargs):
        super(KerasTokenizerConfiguration, self).__init__(**kwargs)
        self.tokenizer_args = tokenizer_args

    @classmethod
    def get_default(cls):
        return cls(tokenizer_args={'oov_token': "UNK"},
                   component_registration_info=RegistrationInfo(tags=['default', 'text'],
                                                                framework='tf',
                                                                namespace='default',
                                                                flag=ComponentFlag.TOKENIZER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(KerasTokenizerConfiguration, self).get_serialization_parameters()
        parameters['tokenizer_args'] = self.tokenizer_args
        return parameters


class TFTextConverterConfiguration(TFConverterConfiguration):

    def __init__(self, max_tokens_limit=None, checkpoint=None, **kwargs):
        super(TFTextConverterConfiguration, self).__init__(**kwargs)
        self.max_tokens_limit = max_tokens_limit
        self.checkpoint = checkpoint

    @classmethod
    def get_default(cls):
        return cls(children=[RegistrationInfo(tags=['default', 'text'],
                                              framework='tf',
                                              namespace='default',
                                              flag=ComponentFlag.TOKENIZER,
                                              internal_key=ProjectRegistry.CONFIGURATION_KEY)],
                   checkpoint=None,
                   feature_registration_info=RegistrationInfo(framework='tf',
                                                              tags=['default', 'text'],
                                                              namespace='default',
                                                              flag=ComponentFlag.FEATURE,
                                                              internal_key=ProjectRegistry.COMPONENT_KEY),
                   component_registration_info=RegistrationInfo(framework='tf',
                                                                tags=['default', 'text'],
                                                                namespace='default',
                                                                flag=ComponentFlag.CONVERTER,
                                                                internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(TFTextConverterConfiguration, self).get_serialization_parameters()
        parameters['max_tokens_limit'] = self.max_tokens_limit
        return parameters
