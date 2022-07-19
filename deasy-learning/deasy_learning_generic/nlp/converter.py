from deasy_learning_generic.converter import BaseConverter
from deasy_learning_generic.registry import ComponentFlag


class TextBaseConverter(BaseConverter):

    def convert_data(self, examples, model_path, label_list, has_labels=True, save_prefix=None, suffix='train'):
        pass

    def __init__(self, max_tokens_limit=None, **kwargs):
        super(TextBaseConverter, self).__init__(**kwargs)
        self.max_tokens_limit = max_tokens_limit if max_tokens_limit else 100000
        assert ComponentFlag.TOKENIZER in self.children