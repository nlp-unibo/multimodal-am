
from deasy_learning_generic.configuration import ProcessorConfiguration
from deasy_learning_generic.registry import RegistrationInfo, ComponentFlag, ProjectRegistry


class TextProcessorConfiguration(ProcessorConfiguration):

    def __init__(self, filter_names=None, retrieve_label=True, disable_filtering=False, **kwargs):
        super(TextProcessorConfiguration, self).__init__(**kwargs)
        self.filter_names = filter_names
        self.disable_filtering = disable_filtering
        self.retrieve_label = retrieve_label

    @classmethod
    def get_default(cls):
        return TextProcessorConfiguration(filter_names=None,
                                          disable_filtering=False,
                                          retrieve_label=True,
                                          component_registration_info=RegistrationInfo(tags=['text', 'default'],
                                                                                       framework='generic',
                                                                                       namespace='default',
                                                                                       flag=ComponentFlag.PROCESSOR,
                                                                                       internal_key=ProjectRegistry.COMPONENT_KEY))

    def get_serialization_parameters(self):
        parameters = super(TextProcessorConfiguration, self).get_serialization_parameters()
        parameters['filter_names'] = self.filter_names
        parameters['disable_filtering'] = self.disable_filtering
        parameters['retrieve_label'] = self.retrieve_label
        return parameters
