

from tqdm import tqdm

from deasy_learning_generic.examples import ExampleList
from deasy_learning_generic.processor import DataProcessor
from deasy_learning_generic.nlp.utility import preprocessing_utils
from deasy_learning_generic.implementations.nlp.examples import TextExample


class TextProcessor(DataProcessor):

    def _get_examples(self, data):
        examples = ExampleList()

        data_keys = data.get_data_keys()
        assert 'text' in data_keys

        total_items = len(data)
        for item_idx in tqdm(range(total_items)):
            item = data[item_idx]

            text = data.get_item_value(item_idx=item_idx, value_key=data_keys['text'])
            text = preprocessing_utils.filter_line(text,
                                                   function_names=self.filter_names,
                                                   disable_filtering=self.disable_filtering)

            if data.has_labels():
                label = self._retrieve_default_label(labels=data.get_labels(), item=item, data_keys=data_keys)
            else:
                label = None

            example = TextExample(text=text, label=label)
            examples.append(example)

        return examples

    def _transform_data(self, data, model_path, suffix, save_prefix=None, component_info=None, filepath=None):
        transformed_data = self._get_examples(data)
        self._save_data(data=transformed_data, model_path=model_path, suffix=suffix,
                        save_prefix=save_prefix, filepath=filepath)

        return transformed_data
