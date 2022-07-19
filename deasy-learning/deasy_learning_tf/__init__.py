"""

Register modules if package is loaded

"""

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_tf.configuration import TFFrameworkHelperConfiguration
from deasy_learning_tf.helper import TFHelper

ProjectRegistry.register_configuration(configuration=TFFrameworkHelperConfiguration.get_default(),
                                       framework='tf',
                                       namespace='default')

ProjectRegistry.register_component(class_type=TFHelper,
                                   flag=ComponentFlag.FRAMEWORK_HELPER,
                                   namespace='default',
                                   framework='tf')


from deasy_learning_tf import implementations