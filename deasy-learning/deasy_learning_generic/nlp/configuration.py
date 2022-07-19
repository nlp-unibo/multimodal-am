from deasy_learning_generic.configuration import Configuration
from deasy_learning_generic.registry import ComponentFlag


class TokenizerConfiguration(Configuration):

    @classmethod
    def get_default(cls) -> 'Configuration':
        pass

    def __init__(self, embedding_dimension=None,
                 embedding_model_type=None, merge_vocabularies=False, **kwargs):
        super(TokenizerConfiguration, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.embedding_model_type = embedding_model_type
        self.merge_vocabularies = merge_vocabularies

    def get_component_flag(self):
        return ComponentFlag.TOKENIZER

    def get_serialization_parameters(self):
        parameters = super(TokenizerConfiguration, self).get_serialization_parameters()
        parameters['embedding_model_type'] = self.embedding_model_type
        return parameters

