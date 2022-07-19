import numpy as np
from functools import partial
from deasy_learning_generic.utility.log_utils import Logger


class NumericInfo(object):

    def __init__(self, name):
        self.name = name
        self.values = []
        self.infos = {}

    def add(self, value):
        self.values.append(value)

    def _operation(self, op_name, op):
        if op_name not in self.infos:
            self.infos[op_name] = op(self.values)
        return self.infos[op_name]

    def average(self):
        return self._operation(op_name='op_average', op=np.mean)

    def quantile(self, q=0.99):
        return self._operation(op_name=f'op_quantile_{q}', op=partial(np.quantile, q=q))

    def max(self):
        return self._operation(op_name='op_max', op=np.max)

    def min(self):
        return self._operation(op_name='op_min', op=np.min)

    def summary(self, keep=False):
        if len(self.values):
            self.average()
            self.quantile(q=0.99)
            self.quantile(q=0.95)
            self.max()
            self.min()

            if not keep:
                self.values.clear()

    def get_info(self, info_name):
        if info_name not in self.infos:
            return 0
        else:
            return self.infos[info_name]

    def show(self):
        for op_name, op_value in self.infos.items():
            Logger.get_logger(__name__).info(f'[{self.name}] Field: {op_name} -- Value: {op_value}')