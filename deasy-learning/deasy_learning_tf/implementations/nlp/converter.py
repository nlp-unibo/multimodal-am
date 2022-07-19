

from tqdm import tqdm

from deasy_learning_generic.examples import ExampleList
from deasy_learning_tf.nlp.converter import TFBaseTextConverter
from deasy_learning_generic.utility.log_utils import Logger

logger = Logger.get_logger(__name__)


class TFTextConverter(TFBaseTextConverter):

    def training_preparation(self, examples, label_list):

        assert isinstance(examples, ExampleList)

        max_seq_length = None
        for example in tqdm(examples):
            features = self.feature_class.convert_example(example=example,
                                                          label_list=label_list,
                                                          converter_info=self.get_info())
            text_ids, label_id = features
            features_ids_len = len(text_ids)

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

        self.label_list = label_list
        self.max_seq_length = min(max_seq_length, self.max_tokens_limit)
