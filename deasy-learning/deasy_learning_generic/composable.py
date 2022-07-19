from abc import ABC
from typing import Dict, AnyStr
import os
import numpy as np
from collections import OrderedDict
import dill

from deasy_learning_generic.utility.log_utils import Logger


class Composable(ABC):

    def __init__(self, children=None, children_args=None):
        children_args = children_args if children_args is not None else {}
        children = children if children is not None else []
        self.children = self._build_children(children=children, children_args=children_args)

    def _build_children(self, children, children_args):
        built_children = {}
        for child_config in children:
            child_flag = child_config.get_component_flag()
            if child_flag in children_args:
                child_additional_args = children_args[child_flag]
            else:
                child_additional_args = None
            child = child_config.retrieve_component(additional_args=child_additional_args)
            built_children[child_flag] = child

        return built_children

    def get_info(self) -> Dict:
        """
        Returns all current instance variables.
        """
        return vars(self)

    def get_filename(self) -> AnyStr:
        """
        Returns the filename for component's state serialization and data serialization.
        This method is abstract and must be implemented by children classes.
        """
        raise NotImplementedError()

    def save_info(self, filepath: AnyStr, prefix: AnyStr = None):
        """
        Serializes component's state.
        The component's state is defined as the set of instance fields (dict).
        Serialization is carried out via Numpy.
        """

        # Create serialization folder if required
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        filename = self.get_filename()

        if prefix is not None and prefix:
            filename = '{0}_{1}.npy'.format(filename, prefix)

        if not filename.endswith('.npy'):
            filename += '.npy'

        # Saving
        filepath = os.path.join(filepath, filename)
        info = self.get_info()

        np.save(filepath, info)

    def load_info(self, filepath: AnyStr, prefix: AnyStr = None):
        """
        Loads serialized component's state.
        """

        assert os.path.isdir(filepath)

        filename = self.get_filename()

        if prefix is not None and prefix:
            filename = '{0}_{1}.npy'.format(filename, prefix)

        if not filename.endswith('.npy'):
            filename += '.npy'

        filepath = os.path.join(filepath, filename)
        if os.path.isfile(filepath):
            Logger.get_logger(__name__).info('[{}] Loading serialized information...'.format(self.__class__.__name__))
            loaded_info = np.load(filepath, allow_pickle=True).item()
            assert type(loaded_info) in [dict, OrderedDict]
            for key, value in loaded_info.items():
                setattr(self, key, value)

    def show_info(self):
        """
        Shows each current instance field, children components included.
        Large memory consuming fields like matrices are omitted.
        """

        for info_name, info in self.get_info().items():
            if info_name == 'children':
                for child_flag, child in info.items():
                    child.show_info()
            else:
                # If numpy.ndarray show shape
                if type(info) in [np.ndarray]:
                    info = info.shape

                # If dict show the first K elements (K = 100)
                if type(info) in [dict, OrderedDict]:
                    if len(info) > 100:
                        continue
                Logger.get_logger(__name__).info(
                    '[{2}] Field: {0} -- Value: {1}'.format(info_name, info, self.__class__.__name__))