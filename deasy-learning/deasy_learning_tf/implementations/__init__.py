"""

Load components if package is loaded

"""

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_tf.implementations.callbacks import EarlyStopping, PredictionRetriever, TrainingLogger
from deasy_learning_tf.implementations.configuration import TFEarlyStoppingConfiguration, TFPredictionRetrieverConfiguration, \
    TFTrainingLogger
from deasy_learning_tf.implementations.nlp.configuration import KerasTokenizerConfiguration, TFTextConverterConfiguration
from deasy_learning_tf.implementations.nlp.converter import TFTextConverter
from deasy_learning_tf.implementations.nlp.features import TFTextFeatures
from deasy_learning_tf.implementations.nlp.tokenizer import KerasTokenizer
from deasy_learning_tf.implementations.helper import register_framework_helper_configurations

# Configurations

# Tokenizer

ProjectRegistry.register_configuration(configuration=KerasTokenizerConfiguration.get_default(),
                                       framework='tf',
                                       namespace='default',
                                       tags=['default', 'text'])

# Converter

ProjectRegistry.register_configuration(configuration=TFTextConverterConfiguration.get_default(),
                                       framework='tf',
                                       namespace='default',
                                       tags=['default', 'text'])

# Callbacks

ProjectRegistry.register_configuration(configuration=TFEarlyStoppingConfiguration.get_default(),
                                       framework='tf',
                                       namespace='default',
                                       tags=['default', 'early_stopping'])

ProjectRegistry.register_configuration(configuration=TFPredictionRetrieverConfiguration.get_default(),
                                       framework='tf',
                                       namespace='default',
                                       tags=['default', 'prediction_retriever'])

ProjectRegistry.register_configuration(configuration=TFTrainingLogger.get_default(),
                                       framework='tf',
                                       namespace='default',
                                       tags=['default', 'training_logger'])

# Helper

register_framework_helper_configurations()

# Components

# Callbacks

ProjectRegistry.register_component(class_type=EarlyStopping,
                                   framework='tf',
                                   flag=ComponentFlag.CALLBACK,
                                   namespace='default',
                                   tags=['default', 'early_stopping'])
ProjectRegistry.register_component(class_type=PredictionRetriever,
                                   framework='tf',
                                   flag=ComponentFlag.CALLBACK,
                                   namespace='default',
                                   tags=['default', 'prediction_retriever'])
ProjectRegistry.register_component(class_type=TrainingLogger,
                                   flag=ComponentFlag.CALLBACK,
                                   namespace='default',
                                   framework='tf',
                                   tags=['default', 'training_logger'])

# Converter

ProjectRegistry.register_component(class_type=TFTextConverter,
                                   flag=ComponentFlag.CONVERTER,
                                   namespace='default',
                                   framework='tf',
                                   tags=['default', 'text'])

# Feature

ProjectRegistry.register_component(class_type=TFTextFeatures,
                                   flag=ComponentFlag.FEATURE,
                                   namespace='default',
                                   framework='tf',
                                   tags=['default', 'text'])

# Tokenizer

ProjectRegistry.register_component(class_type=KerasTokenizer,
                                   flag=ComponentFlag.TOKENIZER,
                                   namespace='default',
                                   framework='tf',
                                   tags=['default', 'text'])
