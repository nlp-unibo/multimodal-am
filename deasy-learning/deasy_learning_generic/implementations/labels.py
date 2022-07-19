

from deasy_learning_generic.labels import Label


class ClassificationLabel(Label):

    def __init__(self, **kwargs):
        super(ClassificationLabel, self).__init__(**kwargs)
        assert self.values is not None

        for value_idx, value in enumerate(self.values):
            self.label_map.setdefault(value, value_idx)

    def convert_label_value(self, label_value, **kwargs):
        return self.label_map[label_value]


class RegressionLabel(Label):

    def __init__(self, **kwargs):
        super(RegressionLabel, self).__init__(**kwargs)
        assert self.values is None

        self.label_map = None

    def convert_label_value(self, label_value, **kwargs):
        return label_value


class GenerativeLabel(Label):

    def __init__(self, **kwargs):
        super(GenerativeLabel, self).__init__(**kwargs)
        assert self.values is None
        self.label_map = {}

    def set_values(self, values):
        assert type(values) == dict
        self.label_map = values

    def convert_label_value(self, label_value, **kwargs):
        assert 'tokenizer' in kwargs
        tokenized = kwargs['tokenizer'].tokenize(label_value, remove_special_tokens=True)

        return tokenized
