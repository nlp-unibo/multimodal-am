"""

Load components if package is loaded

"""

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_generic.implementations.nlp.processor import TextProcessor
from deasy_learning_generic.implementations.nlp.configuration import TextProcessorConfiguration
from deasy_learning_generic.implementations.evaluation import register_evaluation_configurations

# Configurations

ProjectRegistry.register_configuration(configuration=TextProcessorConfiguration.get_default(),
                                       framework='generic',
                                       tags=['default', 'text'],
                                       namespace='default')

# Evaluation

register_evaluation_configurations()

# Components


# Processor

ProjectRegistry.register_component(class_type=TextProcessor, framework='generic', tags=['default', 'text'],
                                   namespace='default', flag=ComponentFlag.PROCESSOR)

