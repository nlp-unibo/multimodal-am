
from deasy_learning_generic.configuration import HelperConfiguration, ConverterConfiguration
from deasy_learning_generic.registry import RegistrationInfo, ComponentFlag, ProjectRegistry


class TFFrameworkHelperConfiguration(HelperConfiguration):

    def __init__(self, strategy, strategy_args, eager_execution=False,
                 limit_gpu_visibility=False, gpu_start_index=None, gpu_end_index=None, **kwargs):
        super(TFFrameworkHelperConfiguration, self).__init__(**kwargs)
        self.eager_execution = eager_execution
        self.strategy = strategy
        self.strategy_args = strategy_args

        self.limit_gpu_visibility = limit_gpu_visibility
        self.gpu_start_index = gpu_start_index
        self.gpu_end_index = gpu_end_index

    @classmethod
    def get_default(cls):
        return TFFrameworkHelperConfiguration(strategy="OneDeviceStrategy",
                                              strategy_args={
                                                  'device': '/gpu:0'
                                              },
                                              component_registration_info=RegistrationInfo(framework='tf',
                                                                                           namespace='default',
                                                                                           flag=ComponentFlag.FRAMEWORK_HELPER,
                                                                                           internal_key=ProjectRegistry.COMPONENT_KEY))


class TFConverterConfiguration(ConverterConfiguration):

    def __init__(self, shuffle_amount=1000, prefetch_amount=100,
                 reshuffle_each_iteration=True, **kwargs):
        super(TFConverterConfiguration, self).__init__(**kwargs)
        self.shuffle_amount = shuffle_amount
        self.prefetch_amount = prefetch_amount
        self.reshuffle_each_iteration = reshuffle_each_iteration
