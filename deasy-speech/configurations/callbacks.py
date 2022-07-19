from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag


def register_arg_aaai_callback_configurations():
    default_es = ProjectRegistry.retrieve_configurations(flag=ComponentFlag.CALLBACK,
                                                         tags=['default', 'early_stopping'],
                                                         namespace='default',
                                                         framework='tf')

    es_f1 = default_es.get_delta_copy(monitor='val_sentence_binary_F1', mode='max')
    ProjectRegistry.register_configuration(configuration=es_f1,
                                           tags=['early_stopping', 'f1'],
                                           namespace='arg_aaai',
                                           framework='tf')

    es_acc = default_es.get_delta_copy(monitor='val_sentence_accuracy', mode='max')
    ProjectRegistry.register_configuration(configuration=es_acc,
                                           tags=['early_stopping', 'accuracy'],
                                           namespace='arg_aaai',
                                           framework='tf')


def register_m_arg_callback_configurations():
    default_es = ProjectRegistry.retrieve_configurations(flag=ComponentFlag.CALLBACK,
                                                         tags=['default', 'early_stopping'],
                                                         namespace='default',
                                                         framework='tf')

    es_f1 = default_es.get_delta_copy(monitor='val_relation_macro_F1', mode='max')
    ProjectRegistry.register_configuration(configuration=es_f1,
                                           tags=['early_stopping', 'f1'],
                                           namespace='m_arg',
                                           framework='tf')

    es_acc = default_es.get_delta_copy(monitor='val_relation_accuracy', mode='max')
    ProjectRegistry.register_configuration(configuration=es_acc,
                                           tags=['early_stopping', 'accuracy'],
                                           namespace='m_arg',
                                           framework='tf')


def register_callback_configurations():
    register_arg_aaai_callback_configurations()
    register_m_arg_callback_configurations()
