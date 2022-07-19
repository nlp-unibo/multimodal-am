import numpy as np
from deasy_learning_generic.utility.python_utils import merge


def _compute_metrics_error(metrics, true_values, predicted_values, prefix=None, label_suffix=None):
    """
    Computes each given metric value, given true and predicted values.

    :param metrics: list of metric functions (typically sci-kit learn metrics)
    :param true_values: ground-truth values
    :param predicted_values: model predicted values
    :return: dict as follows:

        key: metric.__name__
        value: computed metric value
    """

    fold_error_info = {}

    if type(predicted_values) == np.ndarray and type(true_values) == np.ndarray:
        if len(true_values.shape) > 1 and true_values.shape[1] > 1 and len(np.where(true_values[0])[0]) == 1:
            true_values = np.argmax(true_values, axis=1)
            predicted_values = np.argmax(predicted_values, axis=1)

        true_values = true_values.ravel()
        predicted_values = predicted_values.ravel()

    for metric in metrics:
        signal_error = metric(y_true=true_values, y_pred=predicted_values)
        metric_name = metric.name

        if label_suffix is not None:
            metric_name = '{0}_{1}'.format(label_suffix, metric_name)
        if prefix is not None:
            metric_name = '{0}_{1}'.format(prefix, metric_name)

        fold_error_info.setdefault(metric_name, signal_error)

    return fold_error_info


def compute_metrics_error(metrics, true_values, predicted_values, prefix=None):
    """
    Computes each given metric value, given true and predicted values.

    :param metrics: list of metric functions (typically sci-kit learn metrics)
    :param true_values: ground-truth values
    :param predicted_values: model predicted values
    :param prefix:
    :return: dict as follows:

        key: metric.__name__
        value: computed metric value
    """

    fold_error_info = {}

    for key, true_value_set in true_values.items():
        key_metrics = metrics.get_metrics(key)
        pred_value_set = predicted_values[key]
        if type(pred_value_set) == np.ndarray:
            pred_value_set = np.reshape(pred_value_set, true_value_set.shape)
        key_error_info = _compute_metrics_error(metrics=key_metrics, true_values=true_value_set,
                                                predicted_values=pred_value_set, prefix=prefix,
                                                label_suffix=key)
        fold_error_info = merge(fold_error_info, key_error_info)

    return fold_error_info
