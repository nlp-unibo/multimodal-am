import abc


def parse_and_create_filter(data, attribute_name=None):
    if data is None:
        return []

    if isinstance(data, list) and isinstance(data[0], GenericFilter):
        return data

    assert attribute_name is not None
    return [EqualityFilter(attribute_name=attribute_name, reference_value=data)]


class GenericFilter(abc.ABC):

    @abc.abstractmethod
    def __call__(self, element):
        raise NotImplementedError()


class PassFilter(GenericFilter):

    def __init__(self, return_value=True):
        self.return_value = return_value

    def __call__(self, element):
        return self.return_value


class SearchFilter(GenericFilter):

    def __init__(self, attribute_name, reference_value):
        self.attribute_name = attribute_name
        self.reference_value = reference_value

    @abc.abstractmethod
    def __call__(self, element):
        raise NotImplementedError()

    def __repr__(self):
        return f'Attribute: {self.attribute_name} -- Reference: {self.reference_value}'


class EqualityFilter(SearchFilter):

    def __call__(self, element):
        value = getattr(element, self.attribute_name)
        return value == self.reference_value


class InequalityFilter(SearchFilter):

    def __call__(self, element):
        value = getattr(element, self.attribute_name)
        return value != self.reference_value


class MatchFilter(SearchFilter):

    def __init__(self, attribute_name, reference_value):
        super(MatchFilter, self).__init__(attribute_name=attribute_name, reference_value=reference_value)
        assert type(self.reference_value) in [list, set], \
            f"MatchFilter only supports list or set attributes. Got {type(self.reference_value)}"

    def __call__(self, element):
        value = getattr(element, self.attribute_name)
        assert type(value) in [list, set], f"MatchFilter only supports list or set attributes. Got {type(value)}"

        return set(self.reference_value).intersection(set(value)) == set(self.reference_value)


class AntiMatchFilter(MatchFilter):

    def __call__(self, element):
        value = getattr(element, self.attribute_name)
        assert type(value) in [list, set], f"MatchFilter only supports list or set attributes. Got {type(value)}"

        return set(self.reference_value).intersection(set(value)) == set()
