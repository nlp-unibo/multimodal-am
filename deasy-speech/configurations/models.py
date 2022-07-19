from deasy_learning_generic.configuration import ModelConfiguration, ModelParam
from deasy_learning_generic.registry import ComponentFlag, ProjectRegistry, RegistrationInfo


class ArgAAAILSTMConfiguration(ModelConfiguration):

    @classmethod
    def get_default(cls):
        pipeline_configurations = [
            RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                             framework='generic',
                             tags=['default'],
                             namespace='arg_aaai',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY),
            RegistrationInfo(flag=ComponentFlag.CONVERTER,
                             framework='tf',
                             tags=['default'],
                             namespace='arg_aaai',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY)
        ]

        instance = cls(pipeline_configurations=pipeline_configurations,
                       component_registration_info=RegistrationInfo(
                           flag=ComponentFlag.MODEL,
                           tags=['lstm'],
                           namespace='arg_aaai',
                           framework='tf',
                           internal_key=ProjectRegistry.COMPONENT_KEY))

        # Model
        instance.add_param(ModelParam(name='answer_weights', value=[64], flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='l2_regularization', value=0.005, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='optimizer_args', value={
            'beta_2': 0.999,
            'beta_1': 0.9,
            'epsilon': 1e-08,
            'lr': 1e-03
        }, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='lstm_weights', value=[64], flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='tokenizer_args', value={
            'oov_token': 'UNK'
        }, flags=[ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='dropout_rate', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='use_bidirectional', value=True, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='ablation_study', value=None, flags=[ComponentFlag.MODEL]))

        # Pipeline
        instance.add_param(ModelParam(name='filter_names', value=None, flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='normalize_mfccs', value=False, flags=[ComponentFlag.PROCESSOR],
                                      allowed_values=[False, True]))
        instance.add_param(ModelParam(name='pooling_sizes', value=None, flags=[ComponentFlag.PROCESSOR]))

        instance.add_param(ModelParam(name='embedding_dimension', value=100, flags=[ComponentFlag.MODEL,
                                                                                    ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='embedding_model_type', value="glove", flags=[ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='data_mode', value='text_only', flags=[ComponentFlag.PROCESSOR,
                                                                                  ComponentFlag.CONVERTER,
                                                                                  ComponentFlag.MODEL],
                                      allowed_values=['text_only', 'audio_only', 'text_audio']))
        instance.add_param(ModelParam(name='use_audio_features', value=True, flags=[ComponentFlag.DATA_LOADER,
                                                                                    ComponentFlag.PROCESSOR,
                                                                                    ComponentFlag.CONVERTER,
                                                                                    ComponentFlag.MODEL]))
        instance.add_param(
            ModelParam(name='audio_model_name', value='facebook/wav2vec2-base-960h', flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='audio_model_sampling_rate', value=16000, flags=[ComponentFlag.PROCESSOR]))

        # Data Loader
        instance.add_param(ModelParam(name='mode', value='all', flags=[ComponentFlag.DATA_LOADER,
                                                                       ComponentFlag.ROUTINE]))

        return instance


class ArgAAAIBERTConfiguration(ModelConfiguration):

    @classmethod
    def get_default(cls):
        pipeline_configurations = [
            RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                             framework='generic',
                             tags=['default'],
                             namespace='arg_aaai',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY),
            RegistrationInfo(flag=ComponentFlag.CONVERTER,
                             framework='tf',
                             tags=['bert'],
                             namespace='arg_aaai',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY)
        ]

        instance = cls(pipeline_configurations=pipeline_configurations,
                       component_registration_info=RegistrationInfo(
                           flag=ComponentFlag.MODEL,
                           tags=['bert'],
                           namespace='arg_aaai',
                           framework='tf',
                           internal_key=ProjectRegistry.COMPONENT_KEY))

        # Model
        instance.add_param(ModelParam(name='config_args', value={}, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='optimizer_args', value={
            'lr': 5e-05
        }, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='dropout_text', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='answer_units', value=100, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='answer_dropout', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_units', value=64, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_l2', value=0.0005, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='dropout_audio', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_layers',
                                      value=[],
                                      flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='is_bert_trainable',
                                      value=True,
                                      flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='ablation_study', value=None, flags=[ComponentFlag.MODEL]))

        # Pipeline
        instance.add_param(ModelParam(name='preloaded_model_name', value='bert-base-uncased',
                                      flags=[ComponentFlag.MODEL,
                                             ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='normalize_mfccs', value=False, flags=[ComponentFlag.PROCESSOR],
                                      allowed_values=[False, True]))
        instance.add_param(ModelParam(name='pooling_sizes', value=None, flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='data_mode', value='text_only', flags=[ComponentFlag.PROCESSOR,
                                                                                  ComponentFlag.CONVERTER,
                                                                                  ComponentFlag.MODEL],
                                      allowed_values=['text_only', 'audio_only', 'text_audio']))
        instance.add_param(ModelParam(name='use_audio_features', value=True, flags=[ComponentFlag.DATA_LOADER,
                                                                                    ComponentFlag.PROCESSOR,
                                                                                    ComponentFlag.CONVERTER,
                                                                                    ComponentFlag.MODEL]))
        instance.add_param(
            ModelParam(name='audio_model_name', value='facebook/wav2vec2-base-960h', flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='audio_model_sampling_rate', value=16000, flags=[ComponentFlag.PROCESSOR]))

        # Data Loader
        instance.add_param(ModelParam(name='mode', value='all', flags=[ComponentFlag.DATA_LOADER,
                                                                       ComponentFlag.ROUTINE]))

        return instance


def register_arg_aaai_lstm_model_configurations():
    default_config = ArgAAAILSTMConfiguration.get_default()

    # Text
    text_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_only'
    })
    ProjectRegistry.register_configuration(configuration=text_only_config,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['lstm', 'text_only'])

    text_only_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_weights": [128],
        "dropout_rate": 0.5,
        "embedding_dimension": 200,
        "l2_regularization": 0.0005,
        "lstm_weights": [128, 32],
        "optimizer_args": {
            "lr": 0.0001
        }
    })
    ProjectRegistry.register_configuration(configuration=text_only_calibrated,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['lstm', 'text_only', 'calibrated'])

    # Audio
    audio_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'audio_only'
    }, pipeline_configurations=[
        RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                         framework='generic',
                         tags=['default'],
                         namespace='arg_aaai',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY),
        RegistrationInfo(flag=ComponentFlag.CONVERTER,
                         framework='tf',
                         tags=['no_tokenizer'],
                         namespace='arg_aaai',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY)
    ])
    audio_only_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True]
    }, framework='tf', namespace='arg_aaai', tags=['lstm', 'audio_only'])

    audio_only_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [256],
        "dropout_rate": 0.2,
        "l2_regularization": 0.005,
        "lstm_weights": [64, 32],
        "optimizer_args": {
            "lr": 0.001
        },
        "pooling_sizes": [5, 5, 5],
        'use_audio_features': True
    })
    ProjectRegistry.register_configuration(configuration=audio_only_mfccs_calibrated,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['lstm', 'audio_only', "use_audio_features=True", 'calibrated'])

    audio_only_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [32, 32],
        "dropout_rate": 0.5,
        "l2_regularization": 0.001,
        "lstm_weights": [128],
        "optimizer_args": {
            "lr": 0.0002
        },
        'use_audio_features': False
    })
    ProjectRegistry.register_configuration(configuration=audio_only_wav2vec_calibrated,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['lstm', 'audio_only', "use_audio_features=False", 'calibrated'])

    # Text-Audio
    text_audio_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_audio'
    })
    text_audio_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True]
    }, framework='tf', namespace='arg_aaai', tags=['lstm', 'text_audio'])

    text_audio_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [32, 32],
        "dropout_rate": 0.0,
        "embedding_dimension": 300,
        "l2_regularization": 0.0001,
        "lstm_weights": [64, 32],
        "optimizer_args": {
            "lr": 0.001
        },
        "pooling_sizes": [5],
        'use_audio_features': True
    })
    text_audio_mfccs_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='arg_aaai', tags=['lstm', 'text_audio', "use_audio_features=True", 'calibrated'])

    text_audio_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [32, 32],
        "dropout_rate": 0.2,
        "embedding_dimension": 50,
        "l2_regularization": 0.0005,
        "lstm_weights": [64, 64],
        "optimizer_args": {
            "lr": 0.0001
        },
        'use_audio_features': False
    })
    text_audio_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='arg_aaai', tags=['lstm', 'text_audio', "use_audio_features=False", 'calibrated'])


def register_arg_aaai_bert_model_configurations():
    default_config = ArgAAAIBERTConfiguration.get_default()

    # Text
    text_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_only'
    })
    text_only_config.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True]
    }, framework='tf', namespace='arg_aaai', tags=['bert', 'text_only'])

    text_only_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.0,
        "answer_units": 128,
        "dropout_text": 0.0,
    })
    text_only_calibrated.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True]
    }, framework='tf', namespace='arg_aaai', tags=['bert', 'text_only', 'calibrated'])

    # Audio
    audio_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'audio_only'
    }, pipeline_configurations=[
        RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                         framework='generic',
                         tags=['default'],
                         namespace='arg_aaai',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY),
        RegistrationInfo(flag=ComponentFlag.CONVERTER,
                         framework='tf',
                         tags=['no_tokenizer', 'bert'],
                         namespace='arg_aaai',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY)
    ])
    audio_only_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True],
    }, framework='tf', namespace='arg_aaai', tags=['bert', 'audio_only'])

    audio_only_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "audio_l2": 0.0005,
        "audio_layers": [
            {
                "filters": 64,
                "kernel_size": 3,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            }
        ],
        "audio_units": 64,
        "dropout_audio": 0.1,
        "pooling_sizes": [10, 2],
        'use_audio_features': True
    })
    ProjectRegistry.register_configuration(configuration=audio_only_mfccs_calibrated,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['bert', 'audio_only', "use_audio_features=True", 'calibrated'])

    audio_only_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.1,
        "answer_units": 128,
        "audio_l2": 0.0005,
        "audio_layers": [],
        "audio_units": 512,
        "dropout_audio": 0.0,
        'use_audio_features': False
    })
    ProjectRegistry.register_configuration(configuration=audio_only_wav2vec_calibrated,
                                           framework='tf',
                                           namespace='arg_aaai',
                                           tags=['bert', 'audio_only', "use_audio_features=False", 'calibrated'])

    # Text-Audio
    text_audio_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_audio'
    })
    text_audio_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True],
        'is_bert_trainable': [False, True]
    }, framework='tf', namespace='arg_aaai', tags=['bert', 'text_audio'])

    text_audio_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.0,
        "answer_units": 64,
        "audio_l2": 0.0005,
        "audio_layers": [
            {
                "filters": 8,
                "kernel_size": 7,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            },
            {
                "filters": 64,
                "kernel_size": 7,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            }
        ],
        "audio_units": 64,
        "dropout_audio": 0.3,
        "dropout_text": 0.0,
        "pooling_sizes": [2, 2],
        'use_audio_features': True
    })
    text_audio_mfccs_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio'],
        'is_bert_trainable': [False, True]
    }, framework='tf', namespace='arg_aaai', tags=['bert', 'text_audio', "use_audio_features=True", 'calibrated'])

    text_audio_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.0,
        "answer_units": 256,
        "audio_l2": 0.005,
        "audio_layers": [],
        "audio_units": 128,
        "dropout_audio": 0.3,
        "dropout_text": 0.1,
        'use_audio_features': False
    })
    text_audio_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio'],
        'is_bert_trainable': [False, True]
    }, framework='tf', namespace='arg_aaai', tags=['bert', 'text_audio', "use_audio_features=False", 'calibrated'])


def register_arg_aaai_model_configurations():
    register_arg_aaai_lstm_model_configurations()
    register_arg_aaai_bert_model_configurations()


class MArgLSTMConfiguration(ModelConfiguration):

    @classmethod
    def get_default(cls):
        pipeline_configurations = [
            RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                             framework='generic',
                             tags=['default'],
                             namespace='m-arg',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY),
            RegistrationInfo(flag=ComponentFlag.CONVERTER,
                             framework='tf',
                             tags=['default'],
                             namespace='m-arg',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY)
        ]

        instance = cls(pipeline_configurations=pipeline_configurations,
                       component_registration_info=RegistrationInfo(
                           flag=ComponentFlag.MODEL,
                           tags=['lstm'],
                           namespace='m-arg',
                           framework='tf',
                           internal_key=ProjectRegistry.COMPONENT_KEY))

        # Model
        instance.add_param(ModelParam(name='answer_weights', value=[64], flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='l2_regularization', value=0.005, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='optimizer_args', value={
            'beta_2': 0.999,
            'beta_1': 0.9,
            'epsilon': 1e-08,
            'lr': 1e-03
        }, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='lstm_weights', value=[64], flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='tokenizer_args', value={
            'oov_token': 'UNK'
        }, flags=[ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='dropout_rate', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='use_bidirectional', value=True, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='ablation_study', value=None, flags=[ComponentFlag.MODEL]))

        # Pipeline
        instance.add_param(ModelParam(name='filter_names', value=None, flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='normalize_mfccs', value=False, flags=[ComponentFlag.PROCESSOR],
                                      allowed_values=[False, True]))
        instance.add_param(ModelParam(name='pooling_sizes', value=None, flags=[ComponentFlag.PROCESSOR]))

        instance.add_param(ModelParam(name='embedding_dimension', value=100, flags=[ComponentFlag.MODEL,
                                                                                    ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='embedding_model_type', value='glove', flags=[ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='data_mode', value='text_only', flags=[ComponentFlag.PROCESSOR,
                                                                                  ComponentFlag.CONVERTER,
                                                                                  ComponentFlag.MODEL],
                                      allowed_values=['text_only', 'audio_only', 'text_audio']))
        instance.add_param(ModelParam(name='use_audio_features', value=True, flags=[ComponentFlag.DATA_LOADER,
                                                                                    ComponentFlag.PROCESSOR,
                                                                                    ComponentFlag.CONVERTER,
                                                                                    ComponentFlag.MODEL]))
        instance.add_param(
            ModelParam(name='audio_model_name', value='facebook/wav2vec2-base-960h', flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='audio_model_sampling_rate', value=16000, flags=[ComponentFlag.PROCESSOR]))

        # Data Loader
        instance.add_param(ModelParam(name='annotation_confidence', value=0., flags=[ComponentFlag.DATA_LOADER,
                                                                                     ComponentFlag.ROUTINE]))
        instance.add_param(ModelParam(name='mfccs', value=25, flags=[ComponentFlag.DATA_LOADER]))

        return instance


class MArgBERTConfiguration(ModelConfiguration):

    @classmethod
    def get_default(cls):
        pipeline_configurations = [
            RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                             framework='generic',
                             tags=['default'],
                             namespace='m-arg',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY),
            RegistrationInfo(flag=ComponentFlag.CONVERTER,
                             framework='tf',
                             tags=['bert'],
                             namespace='m-arg',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY)
        ]

        instance = cls(pipeline_configurations=pipeline_configurations,
                       component_registration_info=RegistrationInfo(
                           flag=ComponentFlag.MODEL,
                           tags=['bert'],
                           namespace='m-arg',
                           framework='tf',
                           internal_key=ProjectRegistry.COMPONENT_KEY))

        # Model
        instance.add_param(ModelParam(name='config_args', value={}, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='optimizer_args', value={
            'beta_2': 0.999,
            'beta_1': 0.9,
            'epsilon': 1e-08,
            'lr': 5e-05
        }, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='dropout_text', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='answer_units', value=100, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='answer_dropout', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_units', value=128, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_l2', value=0.5, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='dropout_audio', value=0.2, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_layers',
                                      value=[],
                                      flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='is_bert_trainable',
                                      value=False,
                                      flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='ablation_study', value=None, flags=[ComponentFlag.MODEL]))

        # Pipeline
        instance.add_param(ModelParam(name='preloaded_model_name', value='bert-base-uncased',
                                      flags=[ComponentFlag.MODEL,
                                             ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='normalize_mfccs', value=False, flags=[ComponentFlag.PROCESSOR],
                                      allowed_values=[False, True]))
        instance.add_param(ModelParam(name='pooling_sizes', value=None, flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='data_mode', value='text_only', flags=[ComponentFlag.PROCESSOR,
                                                                                  ComponentFlag.CONVERTER,
                                                                                  ComponentFlag.MODEL],
                                      allowed_values=['text_only', 'audio_only', 'text_audio']))
        instance.add_param(ModelParam(name='use_audio_features', value=True, flags=[ComponentFlag.DATA_LOADER,
                                                                                    ComponentFlag.PROCESSOR,
                                                                                    ComponentFlag.CONVERTER,
                                                                                    ComponentFlag.MODEL]))
        instance.add_param(
            ModelParam(name='audio_model_name', value='facebook/wav2vec2-base-960h', flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='audio_model_sampling_rate', value=16000, flags=[ComponentFlag.PROCESSOR]))

        # Data Loader
        instance.add_param(ModelParam(name='annotation_confidence', value=0., flags=[ComponentFlag.DATA_LOADER,
                                                                                     ComponentFlag.ROUTINE]))
        instance.add_param(ModelParam(name='mfccs', value=25, flags=[ComponentFlag.DATA_LOADER]))

        return instance


def register_m_arg_lstm_model_configurations():
    # Default - text only
    default_config = MArgLSTMConfiguration.get_default()

    # Text
    text_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_only'
    })
    text_only_config.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
    }, framework='tf', namespace='m-arg', tags=['lstm', 'text_only'])

    text_only_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_weights": [64],
        "dropout_rate": 0.4,
        "embedding_dimension": 100,
        "l2_regularization": 0.0001,
        "lstm_weights": [128],
        "optimizer_args": {
            "lr": 0.0002
        }
    })
    text_only_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
    }, framework='tf', namespace='m-arg', tags=['lstm', 'text_only', 'calibrated'])

    # Audio
    audio_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'audio_only'
    }, pipeline_configurations=[
        RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                         framework='generic',
                         tags=['default'],
                         namespace='m-arg',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY),
        RegistrationInfo(flag=ComponentFlag.CONVERTER,
                         framework='tf',
                         tags=['no_tokenizer'],
                         namespace='m-arg',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY)
    ])
    audio_only_config.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'use_audio_features': [False, True]
    }, framework='tf', namespace='m-arg', tags=['lstm', 'audio_only'])

    audio_only_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [256],
        "dropout_rate": 0.3,
        "l2_regularization": 0.0001,
        "lstm_weights": [128, 32],
        "optimizer_args": {
            "lr": 0.0001
        },
        "pooling_sizes": [10],
        'use_audio_features': True
    })
    audio_only_mfccs_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
    }, framework='tf', namespace='m-arg', tags=['lstm', 'audio_only', "use_audio_features=True", 'calibrated'])

    audio_only_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [256],
        "dropout_rate": 0.3,
        "l2_regularization": 0.0005,
        "lstm_weights": [64],
        "optimizer_args": {
            "lr": 0.0002
        },
        'use_audio_features': False
    })
    audio_only_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
    }, framework='tf', namespace='m-arg', tags=['lstm', 'audio_only', "use_audio_features=False", 'calibrated'])

    # Text-Audio
    text_audio_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_audio'
    })
    text_audio_config.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'use_audio_features': [False, True]
    }, framework='tf', namespace='m-arg', tags=['lstm', 'text_audio'])

    text_audio_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [64],
        "dropout_rate": 0.0,
        "embedding_dimension": 50,
        "l2_regularization": 0.0001,
        "lstm_weights": [64, 32],
        "optimizer_args": {
            "lr": 0.0001
        },
        "pooling_sizes": [5, 5, 5],
        'use_audio_features': True
    })
    text_audio_mfccs_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='m-arg', tags=['lstm', 'text_audio', "use_audio_features=True", 'calibrated'])

    text_audio_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [256],
        "dropout_rate": 0.0,
        "embedding_dimension": 100,
        "l2_regularization": 0.0001,
        "lstm_weights": [128, 32],
        "optimizer_args": {
            "lr": 0.0002
        },
        'use_audio_features': False
    })
    text_audio_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='m-arg', tags=['lstm', 'text_audio', "use_audio_features=False", 'calibrated'])


def register_m_arg_bert_model_configurations():
    default_config = MArgBERTConfiguration.get_default()

    # Text
    text_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_only'
    })
    text_only_config.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'is_bert_trainable': [False, True],
    }, framework='tf', namespace='m-arg', tags=['bert', 'text_only'])

    text_only_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.5,
        "answer_units": 256,
        "dropout_text": 0.0
    })
    text_only_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'is_bert_trainable': [False, True],
    }, framework='tf', namespace='m-arg', tags=['bert', 'text_only', 'calibrated'])

    # Audio
    audio_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'audio_only'
    }, pipeline_configurations=[
        RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                         framework='generic',
                         tags=['default'],
                         namespace='m-arg',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY),
        RegistrationInfo(flag=ComponentFlag.CONVERTER,
                         framework='tf',
                         tags=['no_tokenizer', 'bert'],
                         namespace='m-arg',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY)
    ])
    audio_only_config.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'use_audio_features': [False, True]
    }, framework='tf', namespace='m-arg', tags=['bert', 'audio_only'])

    audio_only_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.4,
        "answer_units": 64,
        "audio_l2": 0.01,
        "audio_layers": [
            {
                "filters": 8,
                "kernel_size": 7,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            },
            {
                "filters": 64,
                "kernel_size": 7,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            }
        ],
        "audio_units": 512,
        "dropout_audio": 0.2,
        "pooling_sizes": None,
        "use_audio_features": True
    })
    audio_only_mfccs_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
    }, framework='tf', namespace='m-arg', tags=['bert', 'audio_only', "use_audio_features=True", 'calibrated'])

    audio_only_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.3,
        "answer_units": 512,
        "audio_l2": 0.0005,
        "audio_layers": [],
        "audio_units": 128,
        "dropout_audio": 0.0,
        "use_audio_features": False
    })
    audio_only_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
    }, framework='tf', namespace='m-arg', tags=['bert', 'audio_only', "use_audio_features=False", 'calibrated'])

    # Text-Audio
    text_audio_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_audio'
    })
    text_audio_config.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'is_bert_trainable': [False, True],
        'use_audio_features': [False, True]
    }, framework='tf', namespace='m-arg', tags=['bert', 'text_audio'])

    text_audio_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.2,
        "answer_units": 256,
        "audio_l2": 0.005,
        "audio_layers": [
            {
                "filters": 64,
                "kernel_size": 3,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            },
        ],
        "audio_units": 512,
        "dropout_audio": 0.2,
        "dropout_text": 0.0,
        "pooling_sizes": [5, 5, 5],
        "use_audio_features": True
    })
    text_audio_mfccs_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'is_bert_trainable': [False, True],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='m-arg', tags=['bert', 'text_audio', "use_audio_features=True", 'calibrated'])

    text_audio_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.0,
        "answer_units": 100,
        "audio_l2": 0.0001,
        "audio_layers": [],
        "audio_units": 256,
        "dropout_audio": 0.0,
        "dropout_text": 0.0,
        "use_audio_features": False
    })
    text_audio_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'annotation_confidence': [0.00, 0.85],
        'is_bert_trainable': [False, True],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='m-arg', tags=['bert', 'text_audio', "use_audio_features=False", 'calibrated'])


def register_m_arg_model_configurations():
    register_m_arg_lstm_model_configurations()
    register_m_arg_bert_model_configurations()


class UsElecLSTMConfiguration(ModelConfiguration):

    @classmethod
    def get_default(cls):
        pipeline_configurations = [
            RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                             framework='generic',
                             tags=['default'],
                             namespace='us_elec',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY),
            RegistrationInfo(flag=ComponentFlag.CONVERTER,
                             framework='tf',
                             tags=['default'],
                             namespace='us_elec',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY)
        ]

        instance = cls(pipeline_configurations=pipeline_configurations,
                       component_registration_info=RegistrationInfo(
                           flag=ComponentFlag.MODEL,
                           tags=['lstm'],
                           namespace='us_elec',
                           framework='tf',
                           internal_key=ProjectRegistry.COMPONENT_KEY))

        # Model
        instance.add_param(ModelParam(name='answer_weights', value=[64], flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='l2_regularization', value=0.005, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='optimizer_args', value={
            'beta_2': 0.999,
            'beta_1': 0.9,
            'epsilon': 1e-08,
            'lr': 1e-03
        }, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='lstm_weights', value=[64], flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='tokenizer_args', value={
            'oov_token': 'UNK'
        }, flags=[ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='dropout_rate', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='use_bidirectional', value=True, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='ablation_study', value=None, flags=[ComponentFlag.MODEL]))

        # Pipeline
        instance.add_param(ModelParam(name='filter_names', value=None, flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='normalize_mfccs', value=False, flags=[ComponentFlag.PROCESSOR],
                                      allowed_values=[False, True]))
        instance.add_param(ModelParam(name='pooling_sizes', value=None, flags=[ComponentFlag.PROCESSOR]))

        instance.add_param(ModelParam(name='embedding_dimension', value=100, flags=[ComponentFlag.MODEL,
                                                                                    ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='embedding_model_type', value="glove", flags=[ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='data_mode', value='text_only', flags=[ComponentFlag.PROCESSOR,
                                                                                  ComponentFlag.CONVERTER,
                                                                                  ComponentFlag.MODEL,
                                                                                  ComponentFlag.DATA_LOADER],
                                      allowed_values=['text_only', 'audio_only', 'text_audio']))
        instance.add_param(ModelParam(name='use_audio_features', value=True, flags=[ComponentFlag.DATA_LOADER,
                                                                                    ComponentFlag.PROCESSOR,
                                                                                    ComponentFlag.CONVERTER,
                                                                                    ComponentFlag.MODEL]))
        instance.add_param(
            ModelParam(name='audio_model_name', value='facebook/wav2vec2-base-960h', flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='audio_model_sampling_rate', value=16000, flags=[ComponentFlag.PROCESSOR]))

        # Data Loader
        instance.add_param(ModelParam(name='task_type', value='asd', allowed_values=['asd', 'acd'],
                                      flags=[ComponentFlag.DATA_LOADER]))

        return instance


class UsElecBERTConfiguration(ModelConfiguration):

    @classmethod
    def get_default(cls):
        pipeline_configurations = [
            RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                             framework='generic',
                             tags=['default'],
                             namespace='us_elec',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY),
            RegistrationInfo(flag=ComponentFlag.CONVERTER,
                             framework='tf',
                             tags=['bert'],
                             namespace='us_elec',
                             internal_key=ProjectRegistry.CONFIGURATION_KEY)
        ]

        instance = cls(pipeline_configurations=pipeline_configurations,
                       component_registration_info=RegistrationInfo(
                           flag=ComponentFlag.MODEL,
                           tags=['bert'],
                           namespace='us_elec',
                           framework='tf',
                           internal_key=ProjectRegistry.COMPONENT_KEY))

        # Model
        instance.add_param(ModelParam(name='config_args', value={}, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='optimizer_args', value={
            'lr': 5e-05
        }, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='dropout_text', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='answer_units', value=100, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='answer_dropout', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_units', value=64, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_l2', value=0.0005, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='dropout_audio', value=0.1, flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='audio_layers',
                                      value=[],
                                      flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='is_bert_trainable',
                                      value=True,
                                      flags=[ComponentFlag.MODEL]))
        instance.add_param(ModelParam(name='ablation_study', value=None, flags=[ComponentFlag.MODEL]))

        # Pipeline
        instance.add_param(ModelParam(name='preloaded_model_name', value='bert-base-uncased',
                                      flags=[ComponentFlag.MODEL,
                                             ComponentFlag.TOKENIZER]))
        instance.add_param(ModelParam(name='normalize_mfccs', value=False, flags=[ComponentFlag.PROCESSOR],
                                      allowed_values=[False, True]))
        instance.add_param(ModelParam(name='pooling_sizes', value=None, flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='data_mode', value='text_only', flags=[ComponentFlag.PROCESSOR,
                                                                                  ComponentFlag.CONVERTER,
                                                                                  ComponentFlag.MODEL,
                                                                                  ComponentFlag.DATA_LOADER],
                                      allowed_values=['text_only', 'audio_only', 'text_audio']))
        instance.add_param(ModelParam(name='use_audio_features', value=True, flags=[ComponentFlag.DATA_LOADER,
                                                                                    ComponentFlag.PROCESSOR,
                                                                                    ComponentFlag.CONVERTER,
                                                                                    ComponentFlag.MODEL]))
        instance.add_param(
            ModelParam(name='audio_model_name', value='facebook/wav2vec2-base-960h', flags=[ComponentFlag.PROCESSOR]))
        instance.add_param(ModelParam(name='audio_model_sampling_rate', value=16000, flags=[ComponentFlag.PROCESSOR]))

        # Data Loader
        instance.add_param(ModelParam(name='task_type', value='asd', allowed_values=['asd', 'acd'],
                                      flags=[ComponentFlag.DATA_LOADER]))

        return instance


def register_us_elec_lstm_model_configurations():
    default_config = UsElecLSTMConfiguration.get_default()

    # Text
    text_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_only'
    })
    text_only_config.register_combinations_from_params(params_dict={
        'task_type': ['asd', 'acd']
    }, framework='tf', namespace='us_elec', tags=['lstm', 'text_only'])

    # ASD
    text_only_asd_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_weights": [256],
        "dropout_rate": 0.0,
        "embedding_dimension": 100,
        "l2_regularization": 0.0001,
        "lstm_weights": [64, 64],
        "optimizer_args": {
            "lr": 0.0002
        },
        'task_type': 'asd'
    })
    ProjectRegistry.register_configuration(configuration=text_only_asd_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['lstm', 'text_only', 'task_type=asd', 'calibrated'])

    # ACD
    text_only_acd_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_weights": [64],
        "dropout_rate": 0.3,
        "embedding_dimension": 100,
        "l2_regularization": 0.0005,
        "lstm_weights": [64, 32],
        "optimizer_args": {
            "lr": 0.001
        },
        'task_type': 'acd'
    })
    ProjectRegistry.register_configuration(configuration=text_only_acd_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['lstm', 'text_only', 'task_type=acd', 'calibrated'])

    # Audio
    audio_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'audio_only'
    }, pipeline_configurations=[
        RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                         framework='generic',
                         tags=['default'],
                         namespace='us_elec',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY),
        RegistrationInfo(flag=ComponentFlag.CONVERTER,
                         framework='tf',
                         tags=['no_tokenizer'],
                         namespace='us_elec',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY)
    ])
    audio_only_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True],
        'task_type': ['asd', 'acd']
    }, framework='tf', namespace='us_elec', tags=['lstm', 'audio_only'])

    # ASD
    audio_only_asd_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [64],
        "dropout_rate": 0.3,
        "l2_regularization": 0.005,
        "lstm_weights": [64, 32],
        "optimizer_args": {
            "lr": 0.0001
        },
        "pooling_sizes": [5],
        'task_type': 'asd',
        'use_audio_features': True
    })
    ProjectRegistry.register_configuration(configuration=audio_only_asd_mfccs_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['lstm', 'audio_only', 'task_type=asd', "use_audio_features=True",
                                                 'calibrated'])

    audio_only_asd_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [128],
        "dropout_rate": 0.0,
        "l2_regularization": 0.0001,
        "lstm_weights": [32, 32],
        "optimizer_args": {
            "lr": 0.0002
        },
        'task_type': 'asd',
        'use_audio_features': False
    })
    ProjectRegistry.register_configuration(configuration=audio_only_asd_wav2vec_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['lstm', 'audio_only', 'task_type=asd', "use_audio_features=False",
                                                 'calibrated'])

    # ACD
    audio_only_acd_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [64],
        "dropout_rate": 0.1,
        "l2_regularization": 0.0001,
        "lstm_weights": [128, 32],
        "optimizer_args": {
            "lr": 0.0002
        },
        "pooling_sizes": [10],
        'task_type': 'acd',
        'use_audio_features': True
    })
    ProjectRegistry.register_configuration(configuration=audio_only_acd_mfccs_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['lstm', 'audio_only', 'task_type=acd', "use_audio_features=True",
                                                 'calibrated'])

    audio_only_acd_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_weights": [256],
        "dropout_rate": 0.0,
        "l2_regularization": 0.0005,
        "lstm_weights": [128],
        "optimizer_args": {
            "lr": 0.0001
        },
        'task_type': 'acd',
        'use_audio_features': False
    })
    ProjectRegistry.register_configuration(configuration=audio_only_acd_wav2vec_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['lstm', 'audio_only', 'task_type=acd', "use_audio_features=False",
                                                 'calibrated'])

    # Text-Audio
    text_audio_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_audio'
    })
    text_audio_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True],
        'task_type': ['asd', 'acd']
    }, framework='tf', namespace='us_elec', tags=['lstm', 'text_audio'])

    # ASD
    text_audio_asd_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [128],
        "dropout_rate": 0.5,
        "embedding_dimension": 100,
        "l2_regularization": 0.005,
        "lstm_weights": [32, 32],
        "optimizer_args": {
            "lr": 0.0002
        },
        "pooling_sizes": [10, 10],
        'task_type': 'asd',
        'use_audio_features': True
    })
    text_audio_asd_mfccs_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec', tags=['lstm', 'text_audio', 'task_type=asd', "use_audio_features=True", 'calibrated'])

    text_audio_asd_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [32, 32],
        "dropout_rate": 0.3,
        "embedding_dimension": 100,
        "l2_regularization": 0.001,
        "lstm_weights": [64, 32],
        "optimizer_args": {
            "lr": 0.0001
        },
        'task_type': 'asd',
        'use_audio_features': False
    })
    text_audio_asd_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec', tags=['lstm', 'text_audio', 'task_type=asd', "use_audio_features=False", 'calibrated'])

    # ACD
    text_audio_acd_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [32, 32],
        "dropout_rate": 0.1,
        "embedding_dimension": 50,
        "l2_regularization": 0.0005,
        "lstm_weights": [128, 32],
        "optimizer_args": {
            "lr": 0.001
        },
        "pooling_sizes": [10, 10],
        'task_type': 'acd',
        'use_audio_features': True
    })
    text_audio_acd_mfccs_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec', tags=['lstm', 'text_audio', 'task_type=acd', "use_audio_features=True", 'calibrated'])

    text_audio_acd_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_weights": [256],
        "dropout_rate": 0.1,
        "embedding_dimension": 300,
        "l2_regularization": 0.005,
        "lstm_weights": [32, 32],
        "optimizer_args": {
            "lr": 0.0002
        },
        'task_type': 'acd',
        'use_audio_features': False
    })
    text_audio_acd_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec',
        tags=['lstm', 'text_audio', 'task_type=acd', "use_audio_features=False", 'calibrated'])


def register_us_elec_bert_model_configurations():
    default_config = UsElecBERTConfiguration.get_default()

    # Text
    text_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_only',
    })
    text_only_config.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True],
        'task_type': ['asd', 'acd']
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_only'])

    # ASD
    text_only_asd_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.2,
        'answer_units': 128,
        'dropout_text': 0.0,
        'task_type': 'asd'
    })
    text_only_asd_calibrated.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True],
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_only', 'task_type=asd', 'calibrated'])

    # ACD
    text_only_acd_calibrated = text_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.3,
        'answer_units': 100,
        'dropout_text': 0.0,
        'task_type': 'acd'
    })
    text_only_acd_calibrated.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True],
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_only', 'task_type=acd', 'calibrated'])

    # Audio
    audio_only_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'audio_only'
    }, pipeline_configurations=[
        RegistrationInfo(flag=ComponentFlag.PROCESSOR,
                         framework='generic',
                         tags=['default'],
                         namespace='us_elec',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY),
        RegistrationInfo(flag=ComponentFlag.CONVERTER,
                         framework='tf',
                         tags=['no_tokenizer', 'bert'],
                         namespace='us_elec',
                         internal_key=ProjectRegistry.CONFIGURATION_KEY)
    ])
    audio_only_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True],
        'task_type': ['asd', 'acd']
    }, framework='tf', namespace='us_elec', tags=['bert', 'audio_only'])

    # ASD
    audio_only_asd_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.5,
        "answer_units": 512,
        "audio_l2": 0.0001,
        "audio_layers": [
            {
                "filters": 64,
                "kernel_size": 3,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            }
        ],
        "audio_units": 128,
        "dropout_audio": 0.0,
        "pooling_sizes": [5, 5],
        'task_type': 'asd',
        'use_audio_features': True
    })
    ProjectRegistry.register_configuration(configuration=audio_only_asd_mfccs_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['bert', 'audio_only', 'task_type=asd', "use_audio_features=True",
                                                 'calibrated'])

    audio_only_asd_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.3,
        "answer_units": 256,
        "audio_l2": 0.0001,
        "audio_layers": [],
        "audio_units": 100,
        "dropout_audio": 0.3,
        'task_type': 'asd',
        'use_audio_features': False
    })
    ProjectRegistry.register_configuration(configuration=audio_only_asd_wav2vec_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['bert', 'audio_only', 'task_type=asd', "use_audio_features=False",
                                                 'calibrated'])

    # ACD
    audio_only_acd_mfccs_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.2,
        "answer_units": 100,
        "audio_l2": 0.0005,
        "audio_layers": [
            {
                "filters": 64,
                "kernel_size": 3,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            }
        ],
        "audio_units": 512,
        "dropout_audio": 0.2,
        "pooling_sizes": [10, 10],
        'task_type': 'acd',
        'use_audio_features': True
    })
    ProjectRegistry.register_configuration(configuration=audio_only_acd_mfccs_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['bert', 'audio_only', 'task_type=acd', "use_audio_features=True",
                                                 'calibrated'])

    audio_only_acd_wav2vec_calibrated = audio_only_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.4,
        "answer_units": 128,
        "audio_l2": 0.005,
        "audio_layers": [],
        "audio_units": 64,
        "dropout_audio": 0.5,
        'task_type': 'acd',
        'use_audio_features': False
    })
    ProjectRegistry.register_configuration(configuration=audio_only_acd_wav2vec_calibrated,
                                           framework='tf',
                                           namespace='us_elec',
                                           tags=['bert', 'audio_only', 'task_type=acd', "use_audio_features=False",
                                                 'calibrated'])

    # Text-Audio
    text_audio_config = default_config.get_delta_param_copy(params_info={
        'data_mode': 'text_audio'
    })
    text_audio_config.register_combinations_from_params(params_dict={
        'use_audio_features': [False, True],
        'is_bert_trainable': [False, True],
        'task_type': ['asd', 'acd']
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_audio'])

    # ASD
    text_audio_asd_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.4,
        "answer_units": 512,
        "audio_l2": 0.0001,
        "audio_layers": [
            {
                "filters": 64,
                'kernel_size': 3,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            }
        ],
        "audio_units": 512,
        "dropout_audio": 0.3,
        "dropout_text": 0.0,
        'task_type': 'asd',
        'use_audio_features': True,
        'pooling_sizes': [5, 5, 5]
    })
    text_audio_asd_mfccs_calibrated.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_audio', 'task_type=asd', "use_audio_features=True", 'calibrated'])

    text_audio_asd_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.5,
        "answer_units": 128,
        "audio_l2": 0.005,
        "audio_layers": [
        ],
        "audio_units": 100,
        "dropout_audio": 0.1,
        "dropout_text": 0.0,
        'task_type': 'asd',
        'use_audio_features': False,
    })
    text_audio_asd_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_audio', 'task_type=asd', "use_audio_features=False", 'calibrated'])


    # ACD
    text_audio_acd_mfccs_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.1,
        "answer_units": 128,
        "audio_l2": 0.001,
        "audio_layers": [
            {
                "filters": 64,
                'kernel_size': 3,
                "kernel_strides": [1, 1],
                "pool_size": [2, 2],
                "pool_strides": [2, 2]
            }
        ],
        "audio_units": 128,
        "dropout_audio": 0.5,
        "dropout_text": 0.1,
        'task_type': 'acd',
        'use_audio_features': True,
        'pooling_sizes': [10, 10]
    })
    text_audio_acd_mfccs_calibrated.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_audio', 'task_type=acd', "use_audio_features=True", 'calibrated'])

    text_audio_acd_wav2vec_calibrated = text_audio_config.get_delta_param_copy(params_info={
        "answer_dropout": 0.0,
        "answer_units": 100,
        "audio_l2": 0.001,
        "audio_layers": [],
        "audio_units": 256,
        "dropout_audio": 0.3,
        "dropout_text": 0.0,
        'task_type': 'acd',
        'use_audio_features': False
    })
    text_audio_acd_wav2vec_calibrated.register_combinations_from_params(params_dict={
        'is_bert_trainable': [False, True],
        'ablation_study': [None, 'text', 'audio']
    }, framework='tf', namespace='us_elec', tags=['bert', 'text_audio', 'task_type=acd', "use_audio_features=False", 'calibrated'])


def register_us_elec_model_configurations():
    register_us_elec_lstm_model_configurations()
    register_us_elec_bert_model_configurations()


def register_model_configurations():
    register_arg_aaai_model_configurations()
    register_m_arg_model_configurations()
    register_us_elec_model_configurations()
