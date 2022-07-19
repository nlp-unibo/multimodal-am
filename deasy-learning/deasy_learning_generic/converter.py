from typing import AnyStr, Dict

from deasy_learning_generic.component import Component
from deasy_learning_generic.registry import ProjectRegistry


class BaseConverter(Component):

    def _load_data(self, model_path: AnyStr, suffix: AnyStr, component_info: Dict = None, save_prefix: AnyStr = None,
                   filepath: AnyStr = None):
        pass

    def __init__(self, feature_registration_info, **kwargs):
        super(BaseConverter, self).__init__(**kwargs)
        self.feature_class = ProjectRegistry.retrieve_component_from_info(registration_info=feature_registration_info,
                                                                          build=False).class_type

    def _transform_data(self, data, model_path, suffix, save_prefix=None, component_info=None, filepath=None):
        converted_data = self.convert_data(examples=data,
                                           model_path=model_path,
                                           label_list=component_info['labels'],
                                           has_labels=component_info['has_labels'],
                                           suffix=suffix)

        return converted_data

    def get_filename(self):
        return 'converter_info'

    def convert_data(self, examples, model_path, label_list, has_labels=True, save_prefix=None, suffix='train'):
        raise NotImplementedError()

    def convert_example(self, example, label_list, has_labels=True):
        feature = self.feature_class.from_example(example,
                                                  label_list,
                                                  has_labels=has_labels,
                                                  converter_info=self.get_info())
        return feature.to_tensor_format()

    def training_preparation(self, examples, label_list):
        raise NotImplementedError()
