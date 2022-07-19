
import numpy as np
import sklearn.metrics
from sklearn.metrics import f1_score

from deasy_learning_generic.metrics import Metric


class SklearnMetric(Metric):

    def __init__(self, function_name, **kwargs):
        super(SklearnMetric, self).__init__(**kwargs)
        self.function_name = function_name

        assert hasattr(sklearn.metrics, function_name), f'Could not find specified metric in sklearn.metrics module. ' \
                                                        f'Got {function_name}'
        self.metric_method = getattr(sklearn.metrics, self.function_name)

    def __call__(self, y_pred, y_true):
        y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred
        y_true = np.array(y_true) if type(y_true) != np.ndarray else y_true

        return self.metric_method(y_true=y_true, y_pred=y_pred, **self.metric_arguments)


class MaskF1(Metric):

    def __init__(self, **kwargs):
        super(MaskF1, self).__init__(**kwargs)
        assert 'average' in self.metric_arguments
        assert self.metric_arguments['average'] == 'binary'

    def __call__(self, y_pred, y_true):
        y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred
        y_true = np.array(y_true) if type(y_true) != np.ndarray else y_true

        return f1_score(y_true=y_true, y_pred=y_pred, **self.metric_arguments)
