

from deasy_learning_generic.data_loader import DataHandle
import numpy as np
from copy import deepcopy


class DataFrameHandle(DataHandle):

    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray):
            new_instance = deepcopy(self)
            new_instance.data = self.data.iloc[item]
            return new_instance
        if isinstance(item, int):
            return self.data.iloc[item]
        if isinstance(item, slice):
            new_instance = deepcopy(self)
            new_instance.data = self.data[item]
            return new_instance
        else:
            return self.data[item]

    def belongs_to(self, values, target_values):
        return values.isin(target_values).values

    def get_item_value(self, item_idx, value_key):
        return self.data.iloc[item_idx][value_key]