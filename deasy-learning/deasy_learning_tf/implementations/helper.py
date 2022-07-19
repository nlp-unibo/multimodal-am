from deasy_learning_tf.configuration import TFFrameworkHelperConfiguration
from deasy_learning_generic.registry import ProjectRegistry


def register_framework_helper_configurations():
    default_config = TFFrameworkHelperConfiguration.get_default()

    eager_config = default_config.get_delta_copy(eager_execution=True)
    ProjectRegistry.register_configuration(configuration=eager_config,
                                           namespace='default',
                                           tags=['debug', 'eager_execution'],
                                           framework='tf')

    # First gpu
    first_gpu = default_config.get_delta_copy(gpu_start_index=0, gpu_end_index=1, limit_gpu_visibility=True)
    ProjectRegistry.register_configuration(configuration=first_gpu,
                                           framework='tf',
                                           tags=['first_gpu'],
                                           namespace='default')

    first_gpu_eager = first_gpu.get_delta_copy(eager_execution=True)
    ProjectRegistry.register_configuration(configuration=first_gpu_eager,
                                           framework='tf',
                                           tags=['first_gpu', 'eager'],
                                           namespace='default')

    # Second gpu
    second_gpu = default_config.get_delta_copy(gpu_start_index=1, gpu_end_index=2, limit_gpu_visibility=True)
    ProjectRegistry.register_configuration(configuration=second_gpu,
                                           framework='tf',
                                           tags=['second_gpu'],
                                           namespace='default')

    second_gpu_eager = second_gpu.get_delta_copy(eager_execution=True)
    ProjectRegistry.register_configuration(configuration=second_gpu_eager,
                                           framework='tf',
                                           tags=['second_gpu', 'eager'],
                                           namespace='default')
