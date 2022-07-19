from sklearn.utils.class_weight import compute_class_weight

from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.implementations.labels import ClassificationLabel
from deasy_learning_generic.models import Model
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.metric_utils import compute_metrics_error
from deasy_learning_generic.utility.pickle_utils import save_pickle, load_pickle


class SklearnModel(Model):

    def __init__(self, sklearn_model_class, sklearn_model_attributes, **kwargs):
        super(SklearnModel, self).__init__(**kwargs)
        self.sklearn_model_attributes = sklearn_model_attributes
        self.sklearn_model_class = sklearn_model_class
        self.sklearn_model = sklearn_model_class(**self.sklearn_model_attributes)

    # General

    def save(self, filepath, overwrite=True):
        # Model
        save_pickle(filepath=filepath, data=self.sklearn_model)

        # Model state
        model_state = self.get_state()
        if model_state is not None:
            state_path = filepath + '_state.pickle'
            with open(state_path, 'wb') as f:
                save_pickle(filepath=state_path, data=model_state)

    def load(self, filepath, is_external=False, **kwargs):
        # Model
        self.sklearn_model = load_pickle(filepath=filepath)

        # Model state
        state_path = filepath + '_state.pickle'
        self.set_state(model_state=load_pickle(filepath=state_path))

    def predict(self, data, callbacks=None, suffix=DataSplit.TEST, metrics=None):
        callbacks = callbacks or []

        for callback in callbacks:
            if callback.model is None:
                callback.set_model(model=self)
            callback.on_prediction_begin(logs={'suffix': suffix})

        predictions = self.sklearn_model.predict(x=data.get_data())

        for callback in callbacks:
            callback.on_prediction_end(logs={'suffix': suffix})

        if metrics is not None:
            data_labels = data.get_labels()
        else:
            data_labels = None

        ground_truth = self.parse_labels(data_labels)
        metric_results = compute_metrics_error(predicted_values=predictions,
                                               true_values=ground_truth,
                                               prefix=suffix,
                                               metrics=metrics)

        return predictions, metric_results

    def fit(self, train_data, metrics=None, verbose=1, callbacks=None):
        if not hasattr(self.sklearn_model, 'fit'):
            Logger.get_logger(__name__).info(f'{self.sklearn_model_class.__name__} does not support fit! Continuing...')
            return

        callbacks = callbacks or []
        for callback in callbacks:
            callback.set_model(model=self)
            callback.on_train_begin()

        self.sklearn_model.fit(X=train_data.get_data(), y=train_data.get_labels())

        for callback in callbacks:
            callback.on_train_end(logs=None)


class SklearnClassificationModel(SklearnModel):

    def compute_output_weights(self, y_train, label_list):
        self.class_weights = {}
        for label in label_list:
            if isinstance(label, ClassificationLabel):
                label_values = y_train[label.name]
                if len(label_values.shape) > 1:
                    label_values = label_values.ravel()
                label_classes = list(range(label.num_values))
                actual_label_classes = list(set(label_values))
                current_weights = compute_class_weight(class_weight='balanced',
                                                       classes=actual_label_classes, y=label_values)
                remaining_classes = set(label_classes).difference(set(actual_label_classes))

                seen_class_weights = {cls: weight for cls, weight in zip(actual_label_classes, current_weights)}

                for remaining in remaining_classes:
                    seen_class_weights[remaining] = 1.0

                self.class_weights.setdefault(label.name, seen_class_weights)

    def prepare_for_training(self, train_data):
        self.compute_output_weights(y_train=train_data.get_labels(),
                                    label_list=self.label_list)
