from deasy_learning_generic.routines import CVTestRoutine
from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
import os


class ArgAAAIRoutine(CVTestRoutine):

    def __init__(self, mode='all', **kwargs):
        self.mode = mode
        self.folds_path = os.path.join(ProjectRegistry['prebuilt_folds_dir'],
                                       'aaai2016_{}_folds.json'.format(self.mode.lower()))
        kwargs['folds_path'] = self.folds_path
        super(ArgAAAIRoutine, self).__init__(**kwargs)


class MArgRoutine(CVTestRoutine):

    def __init__(self, annotation_confidence=0.00, **kwargs):
        self.annotation_confidence = annotation_confidence
        self.folds_path = os.path.join(ProjectRegistry['prebuilt_folds_dir'],
                                       'm_arg_folds_{:.2f}.json'.format(self.annotation_confidence))

        kwargs['folds_path'] = self.folds_path
        super(MArgRoutine, self).__init__(**kwargs)


def register_routine_components():
    ProjectRegistry.register_component(class_type=ArgAAAIRoutine,
                                       flag=ComponentFlag.ROUTINE,
                                       namespace='arg_aaai',
                                       framework='generic',
                                       tags=['cv_test'])

    ProjectRegistry.register_component(class_type=MArgRoutine,
                                       flag=ComponentFlag.ROUTINE,
                                       namespace='m-arg',
                                       framework='generic',
                                       tags=['cv_test'])
