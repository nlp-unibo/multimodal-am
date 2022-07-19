from abc import ABC


class Metric(ABC):

    def __init__(self, name, metric_arguments):
        self.name = name
        self.metric_arguments = metric_arguments

    def __call__(self, y_pred, y_true):
        raise NotImplementedError()

    def retrieve_parameters_from_network(self, network):
        pass

    def reset(self):
        raise NotImplementedError()

    def __repr__(self):
        return self.name


class MetricManager(ABC):

    def __init__(self, label_metrics_map):
        self.metrics = []
        self.label_metrics_map = label_metrics_map
        self.metric_map = {}

    def add_metric(self, metric):
        self.metrics.append(metric)

    def finalize(self):
        for metric in self.metrics:
            self.metric_map[metric.name] = metric

    def update_metrics_with_model_info(self, model):
        for metric in self.metrics:
            metric.retrieve_parameters_from_network(model)

    def get_metrics(self, label):
        if self.label_metrics_map is not None and len(self.label_metrics_map):
            names = self.label_metrics_map[label]
            return [self.metric_map[name] for name in names]
        else:
            return self.metrics

    def __repr__(self):
        return '--'.join(self.metrics.__repr__())

