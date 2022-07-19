from deasy_learning_generic.configuration import EvaluationConfiguration
from deasy_learning_generic.registry import ProjectRegistry


def register_evaluation_configurations():
    debug_config = EvaluationConfiguration.get_default().get_delta_copy(batch_size=4, epochs=3)
    ProjectRegistry.register_configuration(configuration=debug_config,
                                           namespace='default',
                                           tags=['debug'],
                                           framework='generic')
