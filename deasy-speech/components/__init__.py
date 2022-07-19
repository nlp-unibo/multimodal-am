from components.converter import register_converter_components
from components.features import register_feature_components
from components.data_loader import register_data_loader_components
from components.models import register_model_components
from components.processor import register_processor_components
from components.routines import register_routine_components
from components.tokenizers import register_tokenizer_components

# Data loader
register_data_loader_components()

# Processor
register_processor_components()

# Feature
register_feature_components()

# Tokenizer
register_tokenizer_components()

# Converter
register_converter_components()

# Routines
register_routine_components()

# Model
register_model_components()