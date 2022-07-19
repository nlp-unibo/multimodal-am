from abc import ABC
import os


class Label(ABC):

    def __init__(self, name, values=None):
        self.name = name
        self.label_map = {}
        self.values = values
        self.weights = None

    @property
    def num_values(self):
        return len(self.values) if self.values is not None else None

    def set_values(self, values):
        self.values = values

    def convert_label_value(self, label_value, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.name}--{self.label_map}--{self.values}--{self.weights}'

    def __str__(self):
        return f'{self.name}--{self.label_map}--{self.values}--{self.weights}'


class LabelList(ABC):

    def __init__(self, labels=None):
        self.labels = labels if labels is not None else []
        self.added_state = set()
        self.labels_dict = {}

        if labels is not None:
            for label in labels:
                self.labels_dict.setdefault(label.name, label)

    def __iter__(self):
        return self.labels.__iter__()

    def append(self, label):
        self.labels.append(label)
        if self.labels_dict is not None:
            self.labels_dict.setdefault(label.name, label)

    def __getitem__(self, item):
        return self.labels[item]

    def add_state(self, property_name, property_value):
        setattr(self, property_name, property_value)
        self.added_state.add(property_name)

    def get_state(self, property_name):
        return getattr(self, property_name, None)

    def get_added_state(self):
        return {key: self.get_state(key) for key in self.added_state}

    def as_dict(self):
        for label in self.labels:
            self.labels_dict.setdefault(label.name, label)

        return self.labels_dict

    def get_label_names(self):
        return [label.name for label in self.labels]

    def get_labels_mapping(self):
        return {label.name: label.label_map for label in self.labels}

    def __repr__(self):
        label_representations = ['{}'.format(label) for label in self.labels]
        return os.linesep.join(label_representations)

    def __str__(self):
        label_representations = ['{}'.format(label) for label in self.labels]
        return os.linesep.join(label_representations)

