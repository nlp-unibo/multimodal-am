import os
from abc import ABC
from typing import AnyStr, Any, Dict

from deasy_learning_generic.composable import Composable
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.python_utils import merge
from deasy_learning_generic.data_loader import DataSplit


class Component(Composable, ABC):
    """
    Base component class.
    Each atomic entity in a model pipeline is a component.
    Components generally receive data and produce other data as output.
    This is generally defined as 'data transformation' process.

    """

    def _transform_data(self, data: Any, model_path: AnyStr, suffix: DataSplit,
                        save_prefix: AnyStr = None, component_info: Dict = None,
                        filepath: AnyStr = None) -> Any:
        """
        Applies data transformation process to input data.

        Args:
            data (Any): data to transform derived from a previous component in the pipeline.
            model_path (str): path where to save serialized data.
            suffix (str): it is mainly defined as the data split name (train, val and test).
            save_prefix (str): any specific prefix to be associated with current data. It is mainly derived from routines.
            component_info (dict): any information derived from previous components in the pipeline.
            filepath (str): path where serialized data is stored.

        Returns:
            data (Any): transformed data (if any)

        """

        return data

    def _should_transform(self, component_index: int, model_path: AnyStr, suffix: DataSplit,
                          save_prefix: AnyStr = None, filepath: AnyStr = None,
                          serialized_component_index: int = None, is_child: bool = False) -> bool:
        """
        Checks whether the component should apply the transformation process.

        Args:
            component_index (int): component index with respect to the pipeline (i.e., list of components).
            model_path (str): path where to save serialized data.
            suffix (str): it is mainly defined as the data split name (train, val and test).
            save_prefix (str): any specific prefix to be associated with current data. It is mainly derived from routines.
            filepath (str): path where serialized data is stored.
            serialized_component_index (int): the component index up to which the system has found serialized data.
            is_child (bool): whether the component is a child of a another component.

        Returns:
            Whether the component should transform input data or just pass as it is.

        """

        if serialized_component_index is not None:
            # We skip this component transformation step since later on we load converted data
            if serialized_component_index > component_index:
                return False
            # Check if we have serialized data
            elif component_index == serialized_component_index:
                if is_child:
                    return False
                else:
                    return not self.has_data(model_path=model_path, filepath=filepath,
                                             save_prefix=save_prefix, suffix=suffix)
            # These steps should always apply transformation
            else:
                return True

        return True

    def get_serialized_filepath(self, model_path: AnyStr, suffix: DataSplit, save_prefix: AnyStr = None):
        if save_prefix is not None:
            return os.path.join(model_path, '{0}{1}_data'.format(suffix, save_prefix))
        else:
            return os.path.join(model_path, '{0}_data'.format(suffix))

    def has_data(self, model_path: AnyStr, suffix: DataSplit, save_prefix: AnyStr = None,
                 filepath: AnyStr = None) -> bool:
        """
        Determines whether the current component supports data serialization or not

        Args:
            model_path (str): path where to save serialized data.
            suffix (str): it is mainly defined as the data split name (train, val and test).
            save_prefix (str): any specific prefix to be associated with current data. It is mainly derived from routines.
            filepath (str): path where serialized data is stored.

        Returns:
            True if serialized data exists, False otherwise.
        """

        return os.path.isfile(self.get_serialized_filepath(model_path=model_path,
                                                           save_prefix=save_prefix,
                                                           suffix=suffix))

    def _load_data(self, model_path: AnyStr, suffix: DataSplit, component_info: Dict = None,
                   save_prefix: AnyStr = None, filepath: AnyStr = None):
        """
        Loads serialized transformed data.

        Args:
            model_path (str): path where to save serialized data.
            suffix (str): it is mainly defined as the data split name (train, val and test).
            component_info (dict): a dictionary containing information of previous components in the pipeline.
            save_prefix (str): any specific prefix to be associated with current data. It is mainly derived from routines.
            filepath (str): path where serialized data is stored.
        """
        raise NotImplementedError()

    def apply(self, component_index: int, data: Any, model_path: AnyStr, suffix: DataSplit, save_prefix: AnyStr = None,
              pipeline_info: Dict = None, filepath: AnyStr = None, serialized_component_index: int = None,
              is_child: bool = False, save_info: bool = False, show_info: bool = False) -> (Any, Dict):
        """
        Applies data transformation to input data.
        Children components have execution priority to allow nested calls.

        Transformed data is generally serialized to file disk for speed-up purposes.
        The function checks whether any serialization data exists. If yes, it passes input data as it is
        since the serialized data will be loaded by the next component. Otherwise, data transformation is
        carried out.
        """

        pipeline_info = pipeline_info if pipeline_info is not None else {}

        # Run internal children apply()
        for child_flag, child in self.children.items():
            _, pipeline_info, child_info = child.apply(component_index=component_index, data=data,
                                                       model_path=model_path,
                                                       suffix=suffix, save_prefix=save_prefix,
                                                       pipeline_info=pipeline_info,
                                                       filepath=filepath,
                                                       serialized_component_index=serialized_component_index,
                                                       is_child=True, save_info=save_info)
            pipeline_info[child_flag] = child_info

        # Checking if the component should transform input data or if serialized data already exists
        if self._should_transform(model_path=model_path, suffix=suffix,
                                  save_prefix=save_prefix, filepath=filepath,
                                  serialized_component_index=serialized_component_index,
                                  component_index=component_index, is_child=is_child):
            Logger.get_logger(__name__).info('[{0}] Applying data transformation...'.format(self.__class__.__name__))
            converted_data = self._transform_data(data=data, model_path=model_path, suffix=suffix,
                                                  save_prefix=save_prefix,
                                                  component_info=pipeline_info, filepath=filepath)

            if save_info:
                Logger.get_logger(__name__).info(
                    '[{0}] Saving component information...'.format(self.__class__.__name__))
                self.save_info(filepath=model_path,
                               prefix=save_prefix)

            if show_info:
                self.show_info()
        else:
            if show_info:
                self.show_info()

            # Serialize data if the component supports data serialization
            if self.has_data(model_path=model_path, suffix=suffix, save_prefix=save_prefix,
                             filepath=filepath):
                Logger.get_logger(__name__).info(
                    '[{0}] Attempting to load serialized transformed data...'.format(self.__class__.__name__))
                try:
                    converted_data = self._load_data(model_path=model_path, suffix=suffix, save_prefix=save_prefix,
                                                     filepath=filepath, component_info=pipeline_info)
                except FileNotFoundError:
                    Logger.get_logger(__name__).info('Failed to load serialized data, building new one...')
                    converted_data = self._transform_data(data=data, model_path=model_path, suffix=suffix,
                                                          save_prefix=save_prefix,
                                                          component_info=pipeline_info, filepath=filepath)

                    if save_info:
                        Logger.get_logger(__name__).info(
                            '[{0}] Saving component information...'.format(self.__class__.__name__))
                        self.save_info(filepath=model_path,
                                       prefix=save_prefix)

                    if show_info:
                        self.show_info()

            else:
                converted_data = data

        return converted_data, pipeline_info, self.get_info()

    def compute_stats(self):
        return {}
