from deasy_learning_tf.converter import TFBaseConverter
from deasy_learning_generic.registry import ComponentFlag


class TFBaseTextConverter(TFBaseConverter):

    def __init__(self, max_tokens_limit=None, **kwargs):
        super(TFBaseTextConverter, self).__init__(**kwargs)
        self.max_tokens_limit = max_tokens_limit if max_tokens_limit is not None else 100000

        if len(self.children):
            assert ComponentFlag.TOKENIZER in self.children
