

import tensorflow as tf
from tensorflow.python.keras import backend as K

from deasy_learning_generic.helper import FrameworkHelper


class TFHelper(FrameworkHelper):

    def __init__(self, strategy, strategy_args, shuffle_amount=10000, prefetch_amount=20,
                 reshuffle_each_iteration=True, eager_execution=False,
                 limit_gpu_visibility=False, gpu_start_index=None, gpu_end_index=None, **kwargs):
        super(TFHelper, self).__init__(**kwargs)
        self.eager_execution = eager_execution
        self.strategy = strategy
        self.strategy_args = strategy_args
        self.shuffle_amount = shuffle_amount
        self.prefetch_amount = prefetch_amount
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.limit_gpu_visibility = limit_gpu_visibility
        self.gpu_start_index = gpu_start_index
        self.gpu_end_index = gpu_end_index

    def set_seed(self, seed):
        super(TFHelper, self).set_seed(seed=seed)
        tf.random.set_seed(seed)

    def _limit_gpu_usage(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if self.limit_gpu_visibility:
            assert self.gpu_start_index is not None
            assert self.gpu_end_index is not None
            tf.config.set_visible_devices(gpus[self.gpu_start_index:self.gpu_end_index], "GPU")  # avoid other GPUs
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def _enable_eager_execution(self):
        assert tf.version.VERSION.startswith('2.'), \
            "Tensorflow version is not 2.X! This framework only supports >= 2.0 TF versions"
        tf.config.run_functions_eagerly(self.eager_execution)

    def setup(self):
        self._limit_gpu_usage()
        self._enable_eager_execution()

    def clear_session(self):
        K.clear_session()
