from deasy_learning_generic.configuration import EvaluationConfiguration
from deasy_learning_generic.registry import ProjectRegistry


class ArgAAAIEvaluationConfiguration(EvaluationConfiguration):

    @classmethod
    def get_default(cls):
        return cls(batch_size=16,
                   verbose=1,
                   inference_repetitions=1)


def register_arg_aaai_evaluation_configurations():
    default_config = ArgAAAIEvaluationConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           namespace='arg_aaai',
                                           tags=['default', 'training'],
                                           framework='generic')

    # Debug with batch_size = 1
    low_memory_config = default_config.get_delta_copy(batch_size=1, epochs=1)
    ProjectRegistry.register_configuration(configuration=low_memory_config,
                                           namespace='arg_aaai',
                                           tags=['debug', 'training'],
                                           framework='generic')

    # BERT
    bert_config = default_config.get_delta_copy(batch_size=16)
    ProjectRegistry.register_configuration(configuration=bert_config,
                                           namespace='arg_aaai',
                                           tags=['bert', 'training'],
                                           framework='generic')


class MArgEvaluationConfiguration(EvaluationConfiguration):

    @classmethod
    def get_default(cls):
        return cls(batch_size=16,
                   verbose=1,
                   inference_repetitions=1)


def register_m_arg_evaluation_configurations():
    default_config = MArgEvaluationConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           namespace='m-arg',
                                           tags=['default', 'training'],
                                           framework='generic')

    # Debug with batch_size = 1
    low_memory_config = default_config.get_delta_copy(batch_size=1, epochs=1)
    ProjectRegistry.register_configuration(configuration=low_memory_config,
                                           namespace='m-arg',
                                           tags=['debug', 'training'],
                                           framework='generic')

    # BERT
    bert_config = default_config.get_delta_copy(batch_size=16)
    ProjectRegistry.register_configuration(configuration=bert_config,
                                           namespace='m-arg',
                                           tags=['bert', 'training'],
                                           framework='generic')


class UsElecEvaluationConfiguration(EvaluationConfiguration):

    @classmethod
    def get_default(cls):
        return cls(batch_size=16,
                   verbose=1,
                   inference_repetitions=1)


def register_us_elec_evaluation_configurations():
    default_config = UsElecEvaluationConfiguration.get_default()
    ProjectRegistry.register_configuration(configuration=default_config,
                                           namespace='us_elec',
                                           tags=['default', 'training'],
                                           framework='generic')

    # Debug with batch_size = 1
    low_memory_config = default_config.get_delta_copy(batch_size=16, epochs=1)
    ProjectRegistry.register_configuration(configuration=low_memory_config,
                                           namespace='us_elec',
                                           tags=['debug', 'training'],
                                           framework='generic')

    # BERT
    bert_config = default_config.get_delta_copy(batch_size=16)
    ProjectRegistry.register_configuration(configuration=bert_config,
                                           namespace='us_elec',
                                           tags=['bert', 'training'],
                                           framework='generic')


def register_evaluation_configurations():
    register_arg_aaai_evaluation_configurations()
    register_m_arg_evaluation_configurations()
    register_us_elec_evaluation_configurations()
