

import collections

import tensorflow as tf

from deasy_learning_generic.nlp.tokenizer import BaseTokenizer
from deasy_learning_generic.nlp.utility.embedding_utils import build_embeddings_matrix, load_embedding_model
from deasy_learning_generic.utility.log_utils import Logger


class KerasTokenizer(BaseTokenizer):

    def __init__(self, tokenizer_args=None, **kwargs):
        super(KerasTokenizer, self).__init__(**kwargs)

        tokenizer_args = {} if tokenizer_args is None else tokenizer_args

        assert isinstance(tokenizer_args, dict) or isinstance(tokenizer_args, collections.OrderedDict)

        self.tokenizer_args = tokenizer_args
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)

    def build_vocab(self, data, **kwargs):
        self.tokenizer.fit_on_texts(data)
        self.vocab = self.tokenizer.word_index
        self.vocab_size = len(self.vocab) + 1

        if self.embedding_model_type is not None:
            Logger.get_logger(__name__).info(f'Attempting to load embedding model {self.embedding_model_type},'
                                             f' dimension {self.embedding_dimension}')
            self.embedding_model = load_embedding_model(model_type=self.embedding_model_type,
                                                        embedding_dimension=self.embedding_dimension)

            self.embedding_matrix, self.vocab = build_embeddings_matrix(vocab_size=self.vocab_size,
                                                                        embedding_model=self.embedding_model,
                                                                        embedding_dimension=self.embedding_dimension,
                                                                        word_to_idx=self.vocab)
            self.vocab_size = len(self.vocab) + 1

    def initialize_with_info(self, info):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.word_index = info['vocab']
        self.embedding_model_type = info['embedding_model_type']
        self.embedding_model = info['embedding_model']
        self.embedding_matrix = info['embedding_matrix']

    def tokenize(self, text, remove_special_tokens=False):
        return text

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == str:
            return self.tokenizer.texts_to_sequences([tokens])[0]
        else:
            return self.tokenizer.texts_to_sequences(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.sequences_to_texts(ids)



