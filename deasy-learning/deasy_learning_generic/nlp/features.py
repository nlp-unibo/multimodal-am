from collections import OrderedDict

from deasy_learning_generic.features import Features
from deasy_learning_generic.implementations.labels import ClassificationLabel, RegressionLabel
from deasy_learning_generic.registry import ComponentFlag


class BaseTextFeatures(Features):

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
                    if isinstance(label, ClassificationLabel):
                        label_id.append((label.name, label.convert_label_value(current_example_label)))
                    elif isinstance(label, RegressionLabel):
                        label_id.append((label.name, current_example_label))
                    elif type(current_example_label) == str:
                        assert ComponentFlag.TOKENIZER in converter_info['children']
                        tokenizer = converter_info['children'][ComponentFlag.TOKENIZER]
                        assert tokenizer is not None

                        label_id.append(
                            (label.name, label.convert_label_value(current_example_label, tokenizer=tokenizer)))
                    elif type(current_example_label) == list:
                        label_id.append(
                            (label.name, [label.convert_label_value(item) for item in current_example_label]))
                    else:
                        raise RuntimeError(f'Could not convert given label! Got {label} for example {current_example_label}')

                label_id = OrderedDict(label_id)

        return label_id