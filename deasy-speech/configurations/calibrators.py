from hyperopt import hp

from deasy_learning_generic.configuration import HyperoptCalibratorConfiguration, Configuration
from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag, RegistrationInfo


class LSTMCalibratorConfiguration(HyperoptCalibratorConfiguration):

    def __init__(self, data_mode='text_only', use_audio_features=True, **kwargs):
        super(LSTMCalibratorConfiguration, self).__init__(**kwargs)
        self.data_mode = data_mode
        self.use_audio_features = use_audio_features

        self._exclude_attribute('data_mode')
        self._exclude_attribute('use_audio_features')

        self.search_spaces = {
            'text_only': self._text_only_search_space,
            'audio_only': self._audio_only_search_space,
            'text_audio': self._text_audio_search_space
        }
        self._exclude_attribute('search_spaces')

    @classmethod
    def get_default(cls) -> 'Configuration':
        return cls(component_registration_info=RegistrationInfo(flag=ComponentFlag.CALIBRATOR,
                                                                framework='generic',
                                                                namespace='default',
                                                                tags=['hyperopt'],
                                                                internal_key=ProjectRegistry.COMPONENT_KEY),
                   data_mode='text_only',
                   validate_on='val_binary_F1',
                   validate_condition='maximization',
                   max_evaluations=20)

    def _text_only_search_space(self):
        return {
            'answer_weights': hp.choice('answer_weights', [[64], [128], [256], [32, 32]]),
            'l2_regularization': hp.choice('l2_regularization', [0.005, 0.0005, 0.0001, 0.001, 0.01]),
            'optimizer_args': hp.choice('optimizer_args', [{'lr': 1e-03}, {'lr': 1e-04}, {'lr': 2e-04}]),
            'lstm_weights': hp.choice('lstm_weights', [[64], [32, 32], [128], [64, 32], [64, 64], [128, 32]]),
            'dropout_rate': hp.choice('dropout_rate', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
            'embedding_dimension': hp.choice('embedding_dimension', [50, 100, 200, 300])
        }

    def _audio_only_search_space(self):
        search_space = {
            'answer_weights': hp.choice('answer_weights', [[64], [128], [256], [32, 32]]),
            'l2_regularization': hp.choice('l2_regularization', [0.005, 0.0005, 0.0001, 0.001, 0.01]),
            'optimizer_args': hp.choice('optimizer_args', [{'lr': 1e-03}, {'lr': 1e-04}, {'lr': 2e-04}]),
            'lstm_weights': hp.choice('lstm_weights', [[64], [32, 32], [128], [64, 32], [64, 64], [128, 32]]),
            'dropout_rate': hp.choice('dropout_rate', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
        }

        if self.use_audio_features:
            search_space['pooling_sizes'] = hp.choice('pooling_sizes',
                                       [[5], [2], [10], [2, 2], [5, 2], [10, 2], [5, 5], [5, 5, 5], [10, 10], None])

        return search_space

    def _text_audio_search_space(self):
        search_space = {
            'answer_weights': hp.choice('answer_weights', [[64], [128], [256], [32, 32]]),
            'l2_regularization': hp.choice('l2_regularization', [0.005, 0.0005, 0.0001, 0.001, 0.01]),
            'optimizer_args': hp.choice('optimizer_args', [{'lr': 1e-03}, {'lr': 1e-04}, {'lr': 2e-04}]),
            'lstm_weights': hp.choice('lstm_weights', [[64], [32, 32], [128], [64, 32], [64, 64], [128, 32]]),
            'dropout_rate': hp.choice('dropout_rate', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
            'embedding_dimension': hp.choice('embedding_dimension', [50, 100, 200, 300]),
        }

        if self.use_audio_features:
            search_space['pooling_sizes'] = hp.choice('pooling_sizes',
                                       [[5], [2], [10], [2, 2], [5, 2], [10, 2], [5, 5], [5, 5, 5], [10, 10], None])

        return search_space

    def get_search_space(self):
        return self.search_spaces[self.data_mode]()


def register_dataset_lstm_calibrator_configurations(namespace, validate_on, additional_tags=[]):
    framework = 'tf'

    default_config = LSTMCalibratorConfiguration.get_default().get_delta_copy(
        validate_on=validate_on)

    # Text only
    text_only_params_dict = {
        'data_mode': ['text_only'],
        'use_mongo': [False, True]
    }
    default_config.register_combinations_from_params(param_dict=text_only_params_dict,
                                                     framework=framework,
                                                     namespace=namespace,
                                                     tags=['lstm', 'hyperopt'] + additional_tags)

    # Audio only
    audio_only_params_dict = {
        'data_mode': ['audio_only'],
        'use_audio_features': [False, True],
        'use_mongo': [False, True]
    }
    default_config.register_combinations_from_params(param_dict=audio_only_params_dict,
                                                     framework=framework,
                                                     namespace=namespace,
                                                     tags=['lstm', 'hyperopt'] + additional_tags)

    # Text audio
    text_audio_params_dict = {
        'data_mode': ['text_audio'],
        'use_audio_features': [False, True],
        'use_mongo': [False, True]
    }
    default_config.register_combinations_from_params(param_dict=text_audio_params_dict,
                                                     framework=framework,
                                                     namespace=namespace,
                                                     tags=['lstm', 'hyperopt'] + additional_tags)


def register_arg_aaai_lstm_calibrator_configurations():
    register_dataset_lstm_calibrator_configurations(namespace='arg_aaai',
                                                    validate_on='val_sentence_binary_F1')


def register_marg_baseline_lstm_calibrator_configurations():
    register_dataset_lstm_calibrator_configurations(namespace='m-arg',
                                                    validate_on='val_relation_macro_F1')


def register_us_elec_lstm_calibrator_configurations():
    register_dataset_lstm_calibrator_configurations(namespace='us_elec',
                                                    validate_on='val_sentence_macro_F1',
                                                    additional_tags=['task_type=asd'])

    register_dataset_lstm_calibrator_configurations(namespace='us_elec',
                                                    validate_on='val_component_macro_F1',
                                                    additional_tags=['task_type=acd'])


class BERTCalibratorConfiguration(LSTMCalibratorConfiguration):

    def _text_only_search_space(self):
        return {
            'dropout_text': hp.choice('dropout_text', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
            'answer_units': hp.choice('answer_units', [100, 64, 128, 256, 512]),
            'answer_dropout': hp.choice('answer_dropout', [0., 0.1, 0.2, 0.3, 0.4, 0.5])
        }

    def _audio_only_search_space(self):
        search_space = {
            'audio_units': hp.choice('audio_units', [100, 64, 128, 256, 512]),
            'audio_l2': hp.choice('audio_l2', [0.005, 0.0005, 0.0001, 0.001, 0.01]),
            'dropout_audio': hp.choice('dropout_audio', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
            'answer_units': hp.choice('answer_units', [100, 64, 128, 256, 512]),
            'answer_dropout': hp.choice('answer_dropout', [0., 0.1, 0.2, 0.3, 0.4, 0.5])
        }

        if self.use_audio_features:
            search_space['pooling_sizes'] = hp.choice('pooling_sizes', [[10, 2], [5, 5], [5, 5, 5], [10, 10], None])
            search_space['audio_layers'] = hp.choice("audio_layers", [[{
                'filters': 64,
                'kernel_size': 3,
                'kernel_strides': (1, 1),
                'pool_size': (2, 2),
                'pool_strides': (2, 2)
            }]])
        else:
            search_space['audio_layers'] = hp.choice('audio_layers', [[]])

        return search_space

    def _text_audio_search_space(self):
        search_space = {
            'audio_units': hp.choice('audio_units', [100, 64, 128, 256, 512]),
            'audio_l2': hp.choice('audio_l2', [0.005, 0.0005, 0.0001, 0.001, 0.01]),
            'dropout_audio': hp.choice('dropout_audio', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
            'answer_units': hp.choice('answer_units', [100, 64, 128, 256, 512]),
            'answer_dropout': hp.choice('answer_dropout', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
            'dropout_text': hp.choice('dropout_text', [0., 0.1, 0.2, 0.3, 0.4, 0.5]),
        }

        if self.use_audio_features:
            search_space['pooling_sizes'] = hp.choice('pooling_sizes', [[10, 2], [5, 5], [5, 5, 5], [10, 10], None])
            search_space['audio_layers'] = hp.choice("audio_layers", [[{
                'filters': 64,
                'kernel_size': 3,
                'kernel_strides': (1, 1),
                'pool_size': (2, 2),
                'pool_strides': (2, 2)
            }], [{
                'filters': 8,
                'kernel_size': 7,
                'kernel_strides': (1, 1),
                'pool_size': (2, 2),
                'pool_strides': (2, 2)
            }, {
                'filters': 64,
                'kernel_size': 7,
                'kernel_strides': (1, 1),
                'pool_size': (2, 2),
                'pool_strides': (2, 2)
            }]])
        else:
            search_space['audio_layers'] = hp.choice('audio_layers', [[]])

        return search_space


def register_dataset_bert_calibrator_configurations(namespace, validate_on, additional_tags=[]):
    framework = 'tf'

    default_config = BERTCalibratorConfiguration.get_default().get_delta_copy(validate_on=validate_on)

    # Text only
    text_only_params_dict = {
        'data_mode': ['text_only'],
        'use_mongo': [False, True]
    }
    default_config.register_combinations_from_params(param_dict=text_only_params_dict,
                                                     framework=framework,
                                                     namespace=namespace,
                                                     tags=['bert', 'hyperopt'] + additional_tags)

    # Audio only
    audio_only_params_dict = {
        'data_mode': ['audio_only'],
        'use_audio_features': [False, True],
        'use_mongo': [False, True]
    }
    default_config.register_combinations_from_params(param_dict=audio_only_params_dict,
                                                     framework=framework,
                                                     namespace=namespace,
                                                     tags=['bert', 'hyperopt'] + additional_tags)

    # Text audio
    text_audio_params_dict = {
        'data_mode': ['text_audio'],
        'use_audio_features': [False, True],
        'use_mongo': [False, True]
    }
    default_config.register_combinations_from_params(param_dict=text_audio_params_dict,
                                                     framework=framework,
                                                     namespace=namespace,
                                                     tags=['bert', 'hyperopt'] + additional_tags)


def register_arg_aaai_bert_calibrator_configurations():
    register_dataset_bert_calibrator_configurations(namespace='arg_aaai', validate_on='val_sentence_binary_F1')


def register_marg_bert_calibrator_configurations():
    register_dataset_bert_calibrator_configurations(namespace='m-arg', validate_on='val_relation_macro_F1')


def register_us_elec_bert_calibrator_configurations():
    register_dataset_bert_calibrator_configurations(namespace='us_elec',
                                                    validate_on='val_sentence_macro_F1',
                                                    additional_tags=['task_type=asd'])

    register_dataset_bert_calibrator_configurations(namespace='us_elec',
                                                    validate_on='val_component_macro_F1',
                                                    additional_tags=['task_type=acd'])


def register_calibrator_configurations():
    # Arg AAAI
    register_arg_aaai_lstm_calibrator_configurations()
    register_arg_aaai_bert_calibrator_configurations()

    # M-ARG
    register_marg_baseline_lstm_calibrator_configurations()
    register_marg_bert_calibrator_configurations()

    # MM-UsElecDeb60to16
    register_us_elec_lstm_calibrator_configurations()
    register_us_elec_bert_calibrator_configurations()
