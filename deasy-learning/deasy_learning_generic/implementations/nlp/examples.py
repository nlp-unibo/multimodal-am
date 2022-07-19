

from deasy_learning_generic.examples import Example


class TextExample(Example):

    def __init__(self, text, label=None):
        super(TextExample, self).__init__(label=label)
        self.text = text

    def get_data(self):
        return self.text
