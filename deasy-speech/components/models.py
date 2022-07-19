import tensorflow as tf
from transformers import BertConfig

from components.model_implementations import M_ArgAAAILSTM, \
    M_MArgBERT, M_MArgLSTM, M_ArgAAAIBERT, M_UsElecLSTM, M_UsElecBERT
from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_tf.models import TFClassificationNetwork


class ArgAAAILSTM(TFClassificationNetwork):

    def __init__(self, embedding_dimension, lstm_weights, answer_weights,
                 optimizer_args=None, data_mode='text_only', use_bidirectional=False,
                 l2_regularization=None, dropout_rate=0.2,
                 additional_data=None, use_audio_features=True, ablation_study=None, **kwargs):
        super(ArgAAAILSTM, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.optimizer_args = optimizer_args
        self.use_bidirectional = use_bidirectional
        self.l2_regularization = 0. if l2_regularization is None else l2_regularization
        self.dropout_rate = dropout_rate
        self.data_mode = data_mode
        self.additional_data = additional_data
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

    def build_model(self, pipeline_info):
        self.max_speech_length = pipeline_info[ComponentFlag.CONVERTER]['max_speech_length']
        self.max_speech_length = self.max_speech_length if self.max_speech_length is not None else 0
        self.max_frame_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_length']
        self.max_frame_length = self.max_frame_length if self.max_frame_length is not None else 0
        self.max_frame_features_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_features_length']
        self.max_frame_features_length = self.max_frame_features_length if self.max_frame_features_length is not None else 0

        if self.data_mode != 'audio_only':
            self.vocab_size = pipeline_info[ComponentFlag.TOKENIZER]['vocab_size']
            self.embedding_matrix = pipeline_info[ComponentFlag.TOKENIZER]['embedding_matrix']
        else:
            self.vocab_size = 0
            self.embedding_matrix = None

        self.label_list = pipeline_info[ComponentFlag.CONVERTER]['label_list']

        self.model = M_ArgAAAILSTM(label_list=self.label_list,
                                   embedding_dimension=self.embedding_dimension,
                                   vocab_size=self.vocab_size,
                                   max_speech_length=self.max_speech_length,
                                   max_frame_length=self.max_frame_length,
                                   max_frame_features_length=self.max_frame_features_length,
                                   lstm_weights=self.lstm_weights,
                                   answer_weights=self.answer_weights,
                                   use_bidirectional=self.use_bidirectional,
                                   data_mode=self.data_mode,
                                   l2_regularization=self.l2_regularization,
                                   dropout_rate=self.dropout_rate,
                                   embedding_matrix=self.embedding_matrix,
                                   use_audio_features=self.use_audio_features,
                                   ablation_study=self.ablation_study)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    @tf.function
    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        total_loss, loss_info = self._compute_losses(targets=targets,
                                                     logits=logits,
                                                     label_list=self.label_list)

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info


class ArgAAAIBERT(TFClassificationNetwork):

    def __init__(self, preloaded_model_name, config_args={},
                 optimizer_args=None, data_mode='text_only',
                 dropout_text=0.1, answer_units=100, answer_dropout=0.1,
                 audio_units=128, audio_l2=0.5, dropout_audio=0.2,
                 audio_layers=None, is_bert_trainable=False, additional_data=None,
                 use_audio_features=True, ablation_study=None, **kwargs):
        super(ArgAAAIBERT, self).__init__(**kwargs)
        self.preloaded_model_name = preloaded_model_name
        self.optimizer_args = optimizer_args
        self.data_mode = data_mode
        self.dropout_text = dropout_text
        self.answer_units = answer_units
        self.answer_dropout = answer_dropout
        self.audio_units = audio_units
        self.audio_l2 = audio_l2
        self.dropout_audio = dropout_audio
        self.audio_layers = audio_layers
        self.is_bert_trainable = is_bert_trainable
        self.additional_data = additional_data
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        self.bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=preloaded_model_name)
        # Over-writing config values if required
        for key, value in config_args.items():
            setattr(self.bert_config, key, value)

    def build_model(self, pipeline_info):
        self.max_frame_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_length']
        self.max_frame_length = self.max_frame_length if self.max_frame_length is not None else 0
        self.max_frame_features_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_features_length']
        self.max_frame_features_length = self.max_frame_features_length if self.max_frame_features_length is not None else 0
        self.label_list = pipeline_info[ComponentFlag.CONVERTER]['label_list']

        self.model = M_ArgAAAIBERT(label_list=self.label_list,
                                   max_frame_features_length=self.max_frame_features_length,
                                   max_frame_length=self.max_frame_length,
                                   data_mode=self.data_mode,
                                   bert_config=self.bert_config,
                                   preloaded_model_name=self.preloaded_model_name,
                                   dropout_text=self.dropout_text,
                                   answer_units=self.answer_units,
                                   answer_dropout=self.answer_dropout,
                                   audio_units=self.audio_units,
                                   audio_l2=self.audio_l2,
                                   dropout_audio=self.dropout_audio,
                                   audio_layers=self.audio_layers,
                                   is_bert_trainable=self.is_bert_trainable,
                                   use_audio_features=self.use_audio_features,
                                   ablation_study=self.ablation_study)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        total_loss, loss_info = self._compute_losses(targets=targets,
                                                     logits=logits,
                                                     label_list=self.label_list)

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info


def register_arg_aaai_model_components():
    ProjectRegistry.register_component(class_type=ArgAAAILSTM,
                                       flag=ComponentFlag.MODEL,
                                       framework='tf',
                                       namespace='arg_aaai',
                                       tags=['lstm'])

    ProjectRegistry.register_component(class_type=ArgAAAIBERT,
                                       flag=ComponentFlag.MODEL,
                                       framework='tf',
                                       namespace='arg_aaai',
                                       tags=['bert'])


class MArgLSTM(TFClassificationNetwork):

    def __init__(self, embedding_dimension, lstm_weights, answer_weights,
                 optimizer_args=None, data_mode='text_only', use_bidirectional=False,
                 l2_regularization=None, dropout_rate=0.2,
                 additional_data=None, use_audio_features=True, ablation_study=None, **kwargs):
        super(MArgLSTM, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.optimizer_args = optimizer_args
        self.use_bidirectional = use_bidirectional
        self.l2_regularization = 0. if l2_regularization is None else l2_regularization
        self.dropout_rate = dropout_rate
        self.data_mode = data_mode
        self.additional_data = additional_data
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

    def build_model(self, pipeline_info):
        self.max_speech_length = pipeline_info[ComponentFlag.CONVERTER]['max_speech_length']
        self.max_speech_length = self.max_speech_length if self.max_speech_length is not None else 0
        self.max_frame_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_length']
        self.max_frame_length = self.max_frame_length if self.max_frame_length is not None else 0
        self.max_frame_features_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_features_length']
        self.max_frame_features_length = self.max_frame_features_length if self.max_frame_features_length is not None else 0

        if self.data_mode != 'audio_only':
            self.vocab_size = pipeline_info[ComponentFlag.TOKENIZER]['vocab_size']
            self.embedding_matrix = pipeline_info[ComponentFlag.TOKENIZER]['embedding_matrix']
        else:
            self.vocab_size = 0
            self.embedding_matrix = None

        self.label_list = pipeline_info[ComponentFlag.CONVERTER]['label_list']

        self.model = M_MArgLSTM(label_list=self.label_list,
                                embedding_dimension=self.embedding_dimension,
                                vocab_size=self.vocab_size,
                                max_speech_length=self.max_speech_length,
                                max_frame_length=self.max_frame_length,
                                max_frame_features_length=self.max_frame_features_length,
                                lstm_weights=self.lstm_weights,
                                answer_weights=self.answer_weights,
                                use_bidirectional=self.use_bidirectional,
                                data_mode=self.data_mode,
                                l2_regularization=self.l2_regularization,
                                dropout_rate=self.dropout_rate,
                                embedding_matrix=self.embedding_matrix,
                                use_audio_features=self.use_audio_features,
                                ablation_study=self.ablation_study)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    @tf.function
    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        total_loss, loss_info = self._compute_losses(targets=targets,
                                                     logits=logits,
                                                     label_list=self.label_list)

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info


class MArgBERT(TFClassificationNetwork):

    def __init__(self, preloaded_model_name, config_args={},
                 optimizer_args=None, data_mode='text_only',
                 dropout_text=0.1, answer_units=100, answer_dropout=0.1,
                 audio_units=128, audio_l2=0.5, dropout_audio=0.2,
                 audio_layers=None, is_bert_trainable=False, additional_data=None,
                 use_audio_features=True, ablation_study=None, **kwargs):
        super(MArgBERT, self).__init__(**kwargs)
        self.preloaded_model_name = preloaded_model_name
        self.optimizer_args = optimizer_args
        self.data_mode = data_mode
        self.dropout_text = dropout_text
        self.answer_units = answer_units
        self.answer_dropout = answer_dropout
        self.audio_units = audio_units
        self.audio_l2 = audio_l2
        self.dropout_audio = dropout_audio
        self.audio_layers = audio_layers
        self.is_bert_trainable = is_bert_trainable
        self.additional_data = additional_data
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        self.bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=preloaded_model_name)
        # Over-writing config values if required
        for key, value in config_args.items():
            setattr(self.bert_config, key, value)

    def build_model(self, pipeline_info):
        self.max_frame_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_length']
        self.max_frame_length = self.max_frame_length if self.max_frame_length is not None else 0
        self.max_frame_features_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_features_length']
        self.max_frame_features_length = self.max_frame_features_length if self.max_frame_features_length is not None else 0

        self.label_list = pipeline_info[ComponentFlag.CONVERTER]['label_list']

        self.model = M_MArgBERT(label_list=self.label_list,
                                max_frame_features_length=self.max_frame_features_length,
                                max_frame_length=self.max_frame_length,
                                data_mode=self.data_mode,
                                bert_config=self.bert_config,
                                preloaded_model_name=self.preloaded_model_name,
                                dropout_text=self.dropout_text,
                                answer_units=self.answer_units,
                                answer_dropout=self.answer_dropout,
                                audio_units=self.audio_units,
                                audio_l2=self.audio_l2,
                                dropout_audio=self.dropout_audio,
                                audio_layers=self.audio_layers,
                                is_bert_trainable=self.is_bert_trainable,
                                use_audio_features=self.use_audio_features,
                                ablation_study=self.ablation_study)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        total_loss, loss_info = self._compute_losses(targets=targets,
                                                     logits=logits,
                                                     label_list=self.label_list)

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info


def register_m_arg_model_components():
    ProjectRegistry.register_component(class_type=MArgLSTM,
                                       flag=ComponentFlag.MODEL,
                                       framework='tf',
                                       namespace='m-arg',
                                       tags=['lstm'])

    ProjectRegistry.register_component(class_type=MArgBERT,
                                       flag=ComponentFlag.MODEL,
                                       framework='tf',
                                       namespace='m-arg',
                                       tags=['bert'])


class UsElecLSTM(TFClassificationNetwork):

    def __init__(self, embedding_dimension, lstm_weights, answer_weights,
                 optimizer_args=None, data_mode='text_only', use_bidirectional=False,
                 l2_regularization=None, dropout_rate=0.2,
                 additional_data=None, use_audio_features=True, ablation_study=None, **kwargs):
        super(UsElecLSTM, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.optimizer_args = optimizer_args
        self.use_bidirectional = use_bidirectional
        self.l2_regularization = 0. if l2_regularization is None else l2_regularization
        self.dropout_rate = dropout_rate
        self.data_mode = data_mode
        self.additional_data = additional_data
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

    def build_model(self, pipeline_info):
        self.max_speech_length = pipeline_info[ComponentFlag.CONVERTER]['max_speech_length']
        self.max_speech_length = self.max_speech_length if self.max_speech_length is not None else 0
        self.max_frame_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_length']
        self.max_frame_length = self.max_frame_length if self.max_frame_length is not None else 0
        self.max_frame_features_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_features_length']
        self.max_frame_features_length = self.max_frame_features_length if self.max_frame_features_length is not None else 0

        if self.data_mode != 'audio_only':
            self.vocab_size = pipeline_info[ComponentFlag.TOKENIZER]['vocab_size']
            self.embedding_matrix = pipeline_info[ComponentFlag.TOKENIZER]['embedding_matrix']
        else:
            self.vocab_size = 0
            self.embedding_matrix = None

        self.label_list = pipeline_info[ComponentFlag.CONVERTER]['label_list']

        self.model = M_UsElecLSTM(label_list=self.label_list,
                                  embedding_dimension=self.embedding_dimension,
                                  vocab_size=self.vocab_size,
                                  max_speech_length=self.max_speech_length,
                                  max_frame_length=self.max_frame_length,
                                  max_frame_features_length=self.max_frame_features_length,
                                  lstm_weights=self.lstm_weights,
                                  answer_weights=self.answer_weights,
                                  use_bidirectional=self.use_bidirectional,
                                  data_mode=self.data_mode,
                                  l2_regularization=self.l2_regularization,
                                  dropout_rate=self.dropout_rate,
                                  embedding_matrix=self.embedding_matrix,
                                  use_audio_features=self.use_audio_features,
                                  ablation_study=self.ablation_study)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    @tf.function
    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        total_loss, loss_info = self._compute_losses(targets=targets,
                                                     logits=logits,
                                                     label_list=self.label_list)

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info


class UsElecBERT(TFClassificationNetwork):

    def __init__(self, preloaded_model_name, config_args={},
                 optimizer_args=None, data_mode='text_only',
                 dropout_text=0.1, answer_units=100, answer_dropout=0.1,
                 audio_units=128, audio_l2=0.5, dropout_audio=0.2,
                 audio_layers=None, is_bert_trainable=False, additional_data=None,
                 use_audio_features=True, ablation_study=None, **kwargs):
        super(UsElecBERT, self).__init__(**kwargs)
        self.preloaded_model_name = preloaded_model_name
        self.optimizer_args = optimizer_args
        self.data_mode = data_mode
        self.dropout_text = dropout_text
        self.answer_units = answer_units
        self.answer_dropout = answer_dropout
        self.audio_units = audio_units
        self.audio_l2 = audio_l2
        self.dropout_audio = dropout_audio
        self.audio_layers = audio_layers
        self.is_bert_trainable = is_bert_trainable
        self.additional_data = additional_data
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        self.bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=preloaded_model_name)
        # Over-writing config values if required
        for key, value in config_args.items():
            setattr(self.bert_config, key, value)

    def build_model(self, pipeline_info):
        self.max_frame_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_length']
        self.max_frame_length = self.max_frame_length if self.max_frame_length is not None else 0
        self.max_frame_features_length = pipeline_info[ComponentFlag.CONVERTER]['max_frame_features_length']
        self.max_frame_features_length = self.max_frame_features_length if self.max_frame_features_length is not None else 0
        self.label_list = pipeline_info[ComponentFlag.CONVERTER]['label_list']

        self.model = M_UsElecBERT(label_list=self.label_list,
                                  max_frame_features_length=self.max_frame_features_length,
                                  max_frame_length=self.max_frame_length,
                                  data_mode=self.data_mode,
                                  bert_config=self.bert_config,
                                  preloaded_model_name=self.preloaded_model_name,
                                  dropout_text=self.dropout_text,
                                  answer_units=self.answer_units,
                                  answer_dropout=self.answer_dropout,
                                  audio_units=self.audio_units,
                                  audio_l2=self.audio_l2,
                                  dropout_audio=self.dropout_audio,
                                  audio_layers=self.audio_layers,
                                  is_bert_trainable=self.is_bert_trainable,
                                  use_audio_features=self.use_audio_features,
                                  ablation_study=self.ablation_study)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        total_loss, loss_info = self._compute_losses(targets=targets,
                                                     logits=logits,
                                                     label_list=self.label_list)

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info


def register_us_elec_model_components():
    ProjectRegistry.register_component(class_type=UsElecLSTM,
                                       flag=ComponentFlag.MODEL,
                                       framework='tf',
                                       namespace='us_elec',
                                       tags=['lstm'])

    ProjectRegistry.register_component(class_type=UsElecBERT,
                                       flag=ComponentFlag.MODEL,
                                       framework='tf',
                                       namespace='us_elec',
                                       tags=['bert'])


def register_model_components():
    register_arg_aaai_model_components()
    register_m_arg_model_components()
    register_us_elec_model_components()
