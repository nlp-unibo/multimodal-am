

import numpy as np
from sklearn.metrics import f1_score

from deasy_learning_generic.metrics import Metric
from deasy_learning_generic.nlp.utility.metric_utils import compute_tokens_f1


class TokensF1(Metric):

    def __init__(self, average='macro', **kwargs):
        super(TokensF1, self).__init__(**kwargs)
        self.average = average

    def __call__(self, y_pred, y_true):
        y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred

        y_pred = y_pred.ravel()
        y_true = y_true.ravel()

        pad_indexes = np.where(y_pred == -1)[0]
        sample_weight = np.ones_like(y_pred)
        sample_weight[pad_indexes] = 0.0
        y_pred[pad_indexes] = 0

        return f1_score(y_true=y_pred, y_pred=y_true, sample_weight=sample_weight, average=self.average)


class GeneratedF1(Metric):

    def __init__(self, average='macro', **kwargs):
        super(GeneratedF1, self).__init__(**kwargs)
        self.average = average
        self.vocab = None

    def retrieve_parameters_from_network(self, network):
        assert hasattr(network, 'tokenizer')
        self.vocab = network.tokenizer.vocab

    def __call__(self, y_pred, y_true):
        avg_f1 = []
        for pred, gold in zip(y_pred, y_true):
            avg_f1.append(compute_tokens_f1(a_pred=pred, a_gold=gold))

        return np.mean(avg_f1)

    def reset(self):
        self.vocab = None


