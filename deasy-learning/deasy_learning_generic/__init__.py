"""

Register components if module is loaded

"""

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_generic.configuration import EvaluationConfiguration
from deasy_learning_generic.routines import TrainAndTestRoutine, CVTestRoutine, LooTestRoutine
from deasy_learning_generic.tasks import Task
from deasy_learning_generic.calibrators import HyperOptCalibrator

# Configurations

# Evaluation

ProjectRegistry.register_configuration(configuration=EvaluationConfiguration.get_default(),
                                       framework='generic',
                                       namespace='default',
                                       tags=['default', 'training'])

# Components

# Tasks

ProjectRegistry.register_component(class_type=Task,
                                   framework='generic',
                                   flag=ComponentFlag.TASK,
                                   namespace='default')

# Routines

ProjectRegistry.register_component(class_type=TrainAndTestRoutine,
                                   flag=ComponentFlag.ROUTINE,
                                   framework='generic',
                                   namespace='default',
                                   tags=['train_and_test'])

ProjectRegistry.register_component(class_type=CVTestRoutine,
                                   flag=ComponentFlag.ROUTINE,
                                   framework='generic',
                                   namespace='default',
                                   tags=['cv_test'])

ProjectRegistry.register_component(class_type=LooTestRoutine,
                                   flag=ComponentFlag.ROUTINE,
                                   framework='generic',
                                   namespace='default',
                                   tags=['loo_test'])

# Calibrator
ProjectRegistry.register_component(class_type=HyperOptCalibrator,
                                   flag=ComponentFlag.CALIBRATOR,
                                   framework='generic',
                                   namespace='default',
                                   tags=['hyperopt'])


from deasy_learning_generic import implementations
