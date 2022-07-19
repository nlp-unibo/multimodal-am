from collections import OrderedDict

from deasy_learning_generic.composable import Composable


class Features(Composable):

    def to_tensor_format(self):
        raise NotImplementedError()

    @classmethod
    def get_mappings(cls, converter_info, has_labels=True):
        raise NotImplementedError()

    @classmethod
    def _retrieve_default_label_mappings(cls, mappings, converter_info, has_labels=True):
        raise NotImplementedError()

    @classmethod
    def get_feature_records(cls, feature, converter_info=None):
        raise NotImplementedError()

    @classmethod
    def _retrieve_default_label_feature_records(cls, feature, features, converter_info=None):
        raise NotImplementedError()

    @classmethod
    def get_dataset_selector(cls, label_list):
        raise NotImplementedError()

    @classmethod
    def _retrieve_default_label_dataset_selector(cls, x, record, label_list):
        if label_list is not None:
            labels = [key for key in record if key in label_list.get_label_names()]
            y = {key: record[key] for key in labels}
            return x, y
        else:
            return x

    @classmethod
    def _convert_labels(cls, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_example(cls, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def convert_wrapper_state(cls, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def convert_example(cls, *args, **kwargs):
        raise NotImplementedError()


class BaseFeatures(Features):

    @classmethod
    def _convert_labels(cls, example_label, label_list, has_labels=True, converter_info=None):
        """

        Case: multi-class
            example_label: A, B, C, D
            label_list: [A, B, C, D]
            label_map: {
                A: [0, 0, 0, 1]
                B: [0, 0, 1, 0]
                C: [0, 1, 0, 0]
                D: [1, 0, 0, 0]
            }

        Case: multi-label
            example_label: {
                A: A1,
                B: B2,
                C: C1,
                D: D3
            }
            label_list: {
                A: [A1, A2, A3],
                B: [B1, B2],
                C: [C1, C2, C3],
                D: [D1, D2, D3, D4]
            }
            label_map: {
                A: {
                    A1: [0, 1],
                    A2: [1, 0]
                },
                B: {
                    B1: [0, 0, 1],
                    B2: [0, 1, 0],
                    B3: [1, 0, 0]
                },
                C: { ... },
                D: { ... }
            }

        """

        label_id = None

        if label_list is not None or has_labels:
            label_id = []

            if type(example_label) == list:
                label_id = [OrderedDict([(label.name, label.convert_label_value(ex_label[label.name]))
                                         for label in label_list])
                            for ex_label in example_label]
            else:
                for label in label_list:
                    current_example_label = example_label[label.name]
                    if type(current_example_label) == list:
                        label_id.append(
                            (label.name, [label.convert_label_value(item) for item in current_example_label]))
                    else:
                        label_id.append((label.name, label.convert_label_value(current_example_label)))

                label_id = OrderedDict(label_id)

        return label_id

    @classmethod
    def from_example(cls, example, label_list, converter_info, has_labels=True):
        raise NotImplementedError()

    @classmethod
    def convert_wrapper_state(cls, converter_info=None):
        raise NotImplementedError()

    @classmethod
    def convert_example(cls, example, label_list, has_labels=True, converter_info=None):
        raise NotImplementedError()




