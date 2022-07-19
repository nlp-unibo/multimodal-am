from deasy_learning_generic.nlp.configuration import TokenizerConfiguration
from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag, RegistrationInfo


class BertTokenizerConfiguration(TokenizerConfiguration):

    def __init__(self, preloaded_model_name, **kwargs):
        super(BertTokenizerConfiguration, self).__init__(**kwargs)
        self.preloaded_model_name = preloaded_model_name

    @classmethod
    def get_default(cls):
        return cls(preloaded_model_name='bert-base-uncased',
                   component_registration_info=RegistrationInfo(
                       flag=ComponentFlag.TOKENIZER,
                       framework='generic',
                       tags=['bert'],
                       namespace='transformers',
                       internal_key=ProjectRegistry.COMPONENT_KEY
                   ))

    def get_serialization_parameters(self):
        parameters = super(BertTokenizerConfiguration, self).get_serialization_parameters()
        parameters['preloaded_model_name'] = self.preloaded_model_name
        return parameters


def register_tokenizer_configurations():
    ProjectRegistry.register_configuration(configuration=BertTokenizerConfiguration.get_default(),
                                           framework='generic',
                                           tags=['bert', 'default'],
                                           namespace='transformers')