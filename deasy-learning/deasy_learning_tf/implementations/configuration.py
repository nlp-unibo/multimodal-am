

from deasy_learning_generic.configuration import CallbackConfiguration
from deasy_learning_generic.registry import ComponentFlag, RegistrationInfo, ProjectRegistry


class TFEarlyStoppingConfiguration(CallbackConfiguration):

    def __init__(self, monitor='val_loss', min_delta=0, patience=20,
                 verbose=0, mode='auto', baseline=None, restore_best_weights=True, **kwargs):
        super(TFEarlyStoppingConfiguration, self).__init__(**kwargs)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

    @classmethod
    def get_default(cls):
        return TFEarlyStoppingConfiguration(
            component_registration_info=RegistrationInfo(tags=['default', 'early_stopping'],
                                                         framework='tf',
                                                         namespace='default',
                                                         flag=ComponentFlag.CALLBACK,
                                                         internal_key=ProjectRegistry.COMPONENT_KEY))


class TFPredictionRetrieverConfiguration(CallbackConfiguration):

    @classmethod
    def get_default(cls):
        return TFPredictionRetrieverConfiguration(
            component_registration_info=RegistrationInfo(tags=['default', 'prediction_retriever'],
                                                         flag=ComponentFlag.CALLBACK,
                                                         namespace='default',
                                                         framework='tf',
                                                         internal_key=ProjectRegistry.COMPONENT_KEY))


class TFTrainingLogger(CallbackConfiguration):

    @classmethod
    def get_default(cls):
        return TFTrainingLogger(component_registration_info=RegistrationInfo(tags=['default', 'training_logger'],
                                                                             framework='tf',
                                                                             namespace='default',
                                                                             flag=ComponentFlag.CALLBACK,
                                                                             internal_key=ProjectRegistry.COMPONENT_KEY))
