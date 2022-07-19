from abc import ABC


class Example(ABC):

    def __init__(self, label=None, **kwargs):
        self.label = label
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data(self):
        raise NotImplementedError()


# List wrappers

class ExampleList(ABC):

    def __init__(self):
        self.content = []
        self.added_state = set()

    def __iter__(self):
        return self.content.__iter__()

    def append(self, item):
        self.content.append(item)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        return self.content[item]

    def add_state(self, property_name, property_value):
        setattr(self, property_name, property_value)
        self.added_state.add(property_name)

    def get_state(self, property_name):
        return getattr(self, property_name, None)

    def get_added_state(self):
        return {key: self.get_state(key) for key in self.added_state}

    def get_data(self):
        return [item.get_data() for item in self.content]


