from deasy_learning_generic.composable import Composable
import numpy as np


class FrameworkHelper(Composable):

    def setup(self):
        pass

    def set_seed(self, seed):
        np.random.seed(seed)

    def clear_session(self):
        pass
