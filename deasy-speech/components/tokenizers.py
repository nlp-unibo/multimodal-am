from transformers import BertTokenizer

from deasy_learning_generic.nlp.tokenizer import BaseTokenizer
from deasy_learning_generic.registry import ComponentFlag, ProjectRegistry


class BertTokenizerWrapper(BaseTokenizer):

    def __init__(self, preloaded_model_name, **kwargs):
        super(BertTokenizerWrapper, self).__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(preloaded_model_name)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.preloaded_model_name = preloaded_model_name

    def build_vocab(self, data, **kwargs):
        pass

    # TODO: handle remove_special_tokens
    def tokenize(self, text, remove_special_tokens=False):
        return self.tokenizer.encode_plus(text, truncation=True)

    def get_info(self):
        info = super(BertTokenizerWrapper, self).get_info()
        info['tokenizer'] = self.tokenizer

        return info

    def initialize_with_info(self, info):
        self.tokenizer = BertTokenizer.from_pretrained(self.preloaded_model_name)
        self.tokenizer.word_index = info['vocab']

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids=ids)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

    @property
    def sep_token(self):
        return self.tokenizer.sep_token

    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id


def register_tokenizer_components():
    ProjectRegistry.register_component(class_type=BertTokenizerWrapper,
                                       flag=ComponentFlag.TOKENIZER,
                                       framework='generic',
                                       tags=['bert'],
                                       namespace='transformers')
