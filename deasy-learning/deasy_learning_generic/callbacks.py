from deasy_learning_generic.composable import Composable
from deasy_learning_generic.models import Model
from typing import AnyStr, Dict


class BaseCallback(Composable):
    """
    Generic Callback interface.

    """

    def __init__(self, **kwargs):
        super(BaseCallback, self).__init__(**kwargs)
        self.model = None
        self.save_path = None

    def set_model(self, model: Model):
        """
        Stores the model reference into the callback instance.

        Args:
            model (Model): a model instance.
        """

        self.model = model

    def set_save_path(self, save_path: AnyStr):
        """
        Updates the path to which callback results to be serialized are stored.

        Args:
            save_path (str): path where to save callback's data.
        """

        self.save_path = save_path

    def on_build_model_begin(self, logs: Dict = None):
        """
        Executed before a model instance is created.
        """

        pass

    def on_build_model_end(self, logs: Dict = None):
        """
        Executed after a model instance is created.
        """

        pass

    def on_model_load_begin(self, logs: Dict = None):
        """
        Executed before a model's weights are loaded.
        """

        pass

    def on_model_load_end(self, logs: Dict = None):
        """
        Executed after a model's weights have been loaded.
        """

        pass

    def on_train_begin(self, logs: Dict = None):
        """
        Executed before the training procedure of a routine starts.
        """

        pass

    def on_train_end(self, logs: Dict = None):
        """
        Executed after the training procedure of a routine has ended.
        """

        pass

    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        """
        Executed before the beginning of a training epoch.
        """

        pass

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Executed after a training epoch.
        """

        pass

    def on_batch_begin(self, batch: int, logs: Dict = None):
        """
        Executed before an input batch is fed to a model.
        """
        pass

    def on_batch_end(self, batch: int, logs: Dict = None):
        """
        Executed after an input batch has been fed to a model.
        """
        pass

    def on_prediction_begin(self, logs: Dict = None):
        """
        Executed before the inference procedure of a routine starts.
        """
        pass

    def on_prediction_end(self, logs: Dict = None):
        """
        Executed after the inferences procedure of a routine has ended.
        """
        pass

    def on_batch_prediction_begin(self, batch: int, logs: Dict = None):
        """
        Executed before an input batch for prediction is fed to a model.
        """
        pass

    def on_batch_prediction_end(self, batch: int, logs: Dict = None):
        """
        Executed before an input batch for prediction has been fed to a model.
        """
        pass
