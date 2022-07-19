import tensorflow as tf
from transformers import TFBertModel


class M_ArgAAAILSTM(tf.keras.Model):

    def __init__(self, label_list, embedding_dimension, vocab_size,
                 max_speech_length, max_frame_length, max_frame_features_length,
                 lstm_weights, answer_weights, use_bidirectional=False,
                 data_mode='text_only', l2_regularization=0.,
                 dropout_rate=0.2, embedding_matrix=None, use_audio_features=True,
                 ablation_study=None):
        super(M_ArgAAAILSTM, self).__init__()
        self.text_length = max_speech_length
        self.max_frame_length = max_frame_length
        self.max_frame_features_length = max_frame_features_length
        self.vocab_size = vocab_size
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension
        self.label_list = label_list
        self.data_mode = data_mode
        self.use_bidirectional = use_bidirectional
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        if self.data_mode != 'text_audio':
            self.ablation_study = None

        if self.data_mode != 'audio_only':
            self.input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                             output_dim=embedding_dimension,
                                                             input_length=self.text_length,
                                                             weights=[embedding_matrix] if embedding_matrix is not None
                                                             else embedding_matrix,
                                                             mask_zero=True,
                                                             name='input_embedding')

            self.text_lstm_blocks = []
            for weight_idx, weight in enumerate(self.lstm_weights):
                lstm_block = tf.keras.layers.LSTM(weight,
                                                  return_sequences=False
                                                  if weight_idx == len(self.lstm_weights) - 1 else True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                      self.l2_regularization))

                if self.use_bidirectional:
                    lstm_block = tf.keras.layers.Bidirectional(lstm_block)

                self.text_lstm_blocks.append(lstm_block)
                self.text_lstm_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        if self.data_mode != 'text_only':
            self.audio_lstm_blocks = []
            for weight_idx, weight in enumerate(self.lstm_weights):
                lstm_block = tf.keras.layers.LSTM(weight,
                                                  return_sequences=False
                                                  if weight_idx == len(self.lstm_weights) - 1 else True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                      self.l2_regularization))
                if self.use_bidirectional:
                    lstm_block = tf.keras.layers.Bidirectional(lstm_block)

                self.audio_lstm_blocks.append(lstm_block)
                self.audio_lstm_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.answer_blocks = []
        for weight in answer_weights:
            self.answer_blocks.append(tf.keras.layers.Dense(units=weight,
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_regularizer=tf.keras.regularizers.l2(
                                                                l2_regularization)))
            self.answer_blocks.append(tf.keras.layers.Dropout(rate=dropout_rate))

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              kernel_regularizer=tf.keras.regularizers.l2(
                                                                  l2_regularization),
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def _convert_text(self, x, training=False):
        for block in self.text_lstm_blocks:
            x = block(x, training=training)
        return x

    def _convert_audio(self, x, training=False):
        for block in self.audio_lstm_blocks:
            x = block(x, training=training)
        return x

    def _compute_logits(self, x, training=False):
        for block in self.answer_blocks:
            x = block(x, training=training)
        return x

    def call(self, inputs, training=False, **kwargs):
        stacked_features = None

        # Converting speech text (if any)
        if self.data_mode != 'audio_only':
            # [batch_size, max_speech_length]
            speech_ids = inputs['speech_ids']

            # [batch_size, max_speech_length, emb_dim]
            speech_emb = self.input_embedding(speech_ids)

            # [batch_size, hidden_size]
            speech_emb = self._convert_text(x=speech_emb, training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'text':
                speech_emb = tf.zeros_like(speech_emb)

            stacked_features = speech_emb

        # Converting audio features (if any)
        if self.data_mode != 'text_only':
            # [batch_size, max_frame_length, mfccs]
            if self.use_audio_features:
                speech_mfccs = inputs['speech_mfccs']
                speech_mfccs = tf.reshape(speech_mfccs, [-1, self.max_frame_length, self.max_frame_features_length])

                # [batch_size, audio_hidden_size]
                audio_emb = self._convert_audio(x=speech_mfccs, training=training)
            else:
                speech_audio = inputs['speech_audio']

                # [batch_size, audio_hidden_size]
                audio_emb = self._convert_audio(x=speech_audio[:, None, :], training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'audio':
                audio_emb = tf.zeros_like(audio_emb)

            # [batch_size, hidden_size + audio_hidden_size] <-- audio_only
            # [batch_size, hidden_size * 2 + audio_hidden_size] <-- text_audio
            if stacked_features is None:
                stacked_features = audio_emb
            else:
                stacked_features = tf.concat((stacked_features, audio_emb), axis=-1)

        # Computing answer
        answer = self._compute_logits(x=stacked_features, training=training)
        logits = {key: block(answer, training=training) for key, block in self.final_block.items()}

        return logits, {
            'raw_predictions': {key: tf.nn.softmax(value) for key, value in logits.items()}
        }


class M_ArgAAAIBERT(tf.keras.Model):

    def __init__(self, label_list, bert_config, preloaded_model_name, max_frame_length,
                 max_frame_features_length, dropout_text=0.1, answer_units=100, answer_dropout=0.1,
                 audio_units=128, audio_l2=0.5, dropout_audio=0.2, audio_layers=None,
                 is_bert_trainable=False, data_mode='text_only', use_audio_features=True,
                 ablation_study=None):
        super(M_ArgAAAIBERT, self).__init__()
        self.max_frame_length = max_frame_length
        self.max_frame_features_length = max_frame_features_length
        self.data_mode = data_mode
        self.label_list = label_list
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        if self.data_mode != 'text_audio':
            self.ablation_study = None

        # Text
        if self.data_mode != 'audio_only':
            self.bert = TFBertModel.from_pretrained(pretrained_model_name_or_path=preloaded_model_name,
                                                    config=bert_config, name='bert')
            if not is_bert_trainable:
                for layer in self.bert.layers:
                    layer.trainable = False

            self.text_dropout = tf.keras.layers.Dropout(rate=dropout_text)

        # Audio
        if self.data_mode != 'text_only':
            self.audio_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=audio_units,
                                                                                 return_sequences=True,
                                                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                                                     l2=audio_l2)))
            self.audio_dropout = tf.keras.layers.Dropout(rate=dropout_audio)

            self.audio_cnn_blocks = []
            self.audio_layers = audio_layers if audio_layers is not None else []
            for audio_layer in self.audio_layers:
                audio_cnn = tf.keras.layers.Conv2D(filters=audio_layer['filters'],
                                                   kernel_size=audio_layer['kernel_size'],
                                                   strides=audio_layer['kernel_strides'],
                                                   padding='valid')
                self.audio_cnn_blocks.append(audio_cnn)
                self.audio_cnn_blocks.append(tf.keras.layers.BatchNormalization())
                self.audio_cnn_blocks.append(tf.keras.layers.Activation('relu'))
                self.audio_cnn_blocks.append(tf.keras.layers.Dropout(rate=dropout_audio))
                self.audio_cnn_blocks.append(tf.keras.layers.MaxPool2D(pool_size=audio_layer['pool_size'],
                                                                       strides=audio_layer['pool_strides']))

        # Answer
        self.answer_blocks = [
            tf.keras.layers.Dense(units=answer_units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(rate=answer_dropout)
        ]

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def _convert_text(self, text_ids, text_attention_mask, training=False):
        text_emb = self.bert({
            'input_ids': text_ids,
            'attention_mask': text_attention_mask
        }, training=training)[1]
        text_emb = self.text_dropout(text_emb, training=training)

        return text_emb

    def _convert_audio(self, audio_features, training=False):
        # LSTM embedding
        audio_emb = self.audio_lstm(audio_features, training=training)
        audio_emb = tf.reshape(audio_emb, [audio_emb.shape[0], -1])
        audio_emb = self.audio_dropout(audio_emb, training=training)

        # CNN embedding
        if len(self.audio_cnn_blocks):
            audio_cnn = tf.expand_dims(audio_features, axis=-1)
            for block in self.audio_cnn_blocks:
                audio_cnn = block(audio_cnn, training=training)

            audio_cnn = tf.reshape(audio_cnn, [audio_cnn.shape[0], -1])

            audio_emb = tf.concat((audio_emb, audio_cnn), axis=-1)

        return audio_emb

    def _compute_logits(self, features, training=False):
        answer = features
        for block in self.answer_blocks:
            answer = block(answer, training=training)

        logits = {key: block(answer, training=training) for key, block in self.final_block.items()}
        return logits

    def call(self, inputs, training=False, **kwargs):
        stacked_features = None

        # Converting speech text (if any)
        if self.data_mode != 'audio_only':
            speech_emb = self._convert_text(text_ids=inputs['speech_ids'],
                                            text_attention_mask=inputs['speech_attention_mask'],
                                            training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'text':
                speech_emb = tf.zeros_like(speech_emb)

            stacked_features = speech_emb

        # Converting audio features (if any)
        if self.data_mode != 'text_only':
            if self.use_audio_features:
                # [batch_size, max_frame_length, mfccs]
                speech_mfccs = inputs['speech_mfccs']
                speech_mfccs = tf.reshape(speech_mfccs, [speech_mfccs.shape[0], -1, self.max_frame_features_length])

                audio_emb = self._convert_audio(audio_features=speech_mfccs, training=training)
            else:
                speech_audio = inputs['speech_audio']

                # [batch_size, audio_hidden_size]
                audio_emb = self._convert_audio(audio_features=speech_audio[:, None, :], training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'audio':
                audio_emb = tf.zeros_like(audio_emb)

            if stacked_features is None:
                stacked_features = audio_emb
            else:
                stacked_features = tf.concat((stacked_features, audio_emb), axis=-1)

        logits = self._compute_logits(features=stacked_features, training=training)

        return logits, {
            'raw_predictions': {key: tf.nn.softmax(value) for key, value in logits.items()}
        }


class M_MArgLSTM(tf.keras.Model):

    def __init__(self, label_list, embedding_dimension, vocab_size,
                 max_speech_length, max_frame_length, max_frame_features_length,
                 lstm_weights, answer_weights, use_bidirectional=False,
                 data_mode='text_only', l2_regularization=0.,
                 dropout_rate=0.2, embedding_matrix=None,
                 use_audio_features=True, ablation_study=None):
        super(M_MArgLSTM, self).__init__()
        self.max_speech_length = max_speech_length
        self.max_frame_length = max_frame_length
        self.max_frame_features_length = max_frame_features_length
        self.vocab_size = vocab_size
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension
        self.label_list = label_list
        self.data_mode = data_mode
        self.use_bidirectional = use_bidirectional
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        if self.data_mode != 'text_audio':
            self.ablation_study = None

        if self.data_mode != 'audio_only':
            self.input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                             output_dim=embedding_dimension,
                                                             input_length=self.max_speech_length,
                                                             weights=[embedding_matrix] if embedding_matrix is not None
                                                             else embedding_matrix,
                                                             mask_zero=True,
                                                             name='input_embedding')

            self.text_lstm_blocks = []
            for weight_idx, weight in enumerate(self.lstm_weights):
                lstm_block = tf.keras.layers.LSTM(weight,
                                                  return_sequences=False
                                                  if weight_idx == len(self.lstm_weights) - 1 else True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                      self.l2_regularization))

                if self.use_bidirectional:
                    lstm_block = tf.keras.layers.Bidirectional(lstm_block)

                self.text_lstm_blocks.append(lstm_block)
                self.text_lstm_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        if self.data_mode != 'text_only':
            self.audio_lstm_blocks = []
            for weight_idx, weight in enumerate(self.lstm_weights):
                lstm_block = tf.keras.layers.LSTM(weight,
                                                  return_sequences=False
                                                  if weight_idx == len(self.lstm_weights) - 1 else True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                      self.l2_regularization))
                if self.use_bidirectional:
                    lstm_block = tf.keras.layers.Bidirectional(lstm_block)

                self.audio_lstm_blocks.append(lstm_block)
                self.audio_lstm_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.answer_blocks = []
        for weight in answer_weights:
            self.answer_blocks.append(tf.keras.layers.Dense(units=weight,
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_regularizer=tf.keras.regularizers.l2(
                                                                l2_regularization)))
            self.answer_blocks.append(tf.keras.layers.Dropout(rate=dropout_rate))

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              kernel_regularizer=tf.keras.regularizers.l2(
                                                                  l2_regularization),
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def _convert_text(self, x, training=False):
        for block in self.text_lstm_blocks:
            x = block(x, training=training)
        return x

    def _convert_audio(self, x, training=False):
        for block in self.audio_lstm_blocks:
            x = block(x, training=training)
        return x

    def _compute_logits(self, x, training=False):
        for block in self.answer_blocks:
            x = block(x, training=training)

        logits = {key: block(x, training=training) for key, block in self.final_block.items()}
        return logits

    def call(self, inputs, training=False, **kwargs):
        stacked_features = None

        # Text
        if self.data_mode != 'audio_only':
            text_a_emb = self.input_embedding(inputs['text_a_ids'])
            text_a_emb = self._convert_text(x=text_a_emb,
                                            training=training)

            text_b_emb = self.input_embedding(inputs['text_b_ids'])
            text_b_emb = self._convert_text(x=text_b_emb,
                                            training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'text':
                text_a_emb = tf.zeros_like(text_a_emb)
                text_b_emb = tf.zeros_like(text_b_emb)

            stacked_features = tf.concat((text_a_emb, text_b_emb), axis=-1)

        # Audio
        if self.data_mode != 'text_only':
            if self.use_audio_features:
                audio_a_mfccs = inputs['audio_a_mfccs']
                audio_a_mfccs = tf.reshape(audio_a_mfccs, [audio_a_mfccs.shape[0], -1, self.max_frame_features_length])
                audio_a_emb = self._convert_audio(x=audio_a_mfccs,
                                                  training=training)

                audio_b_mfccs = inputs['audio_b_mfccs']
                audio_b_mfccs = tf.reshape(audio_b_mfccs, [audio_b_mfccs.shape[0], -1, self.max_frame_features_length])
                audio_b_emb = self._convert_audio(x=audio_b_mfccs,
                                                  training=training)
            else:
                audio_a_data = inputs['audio_a_data']
                audio_a_emb = self._convert_audio(x=audio_a_data[:, None, :], training=training)

                audio_b_data = inputs['audio_b_data']
                audio_b_emb = self._convert_audio(x=audio_b_data[:, None, :], training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'audio':
                audio_a_emb = tf.zeros_like(audio_a_emb)
                audio_b_emb = tf.zeros_like(audio_b_emb)

            if self.data_mode == 'audio_only':
                stacked_features = tf.concat((audio_a_emb, audio_b_emb), axis=-1)
            else:
                stacked_features = tf.concat((stacked_features, audio_a_emb, audio_b_emb), axis=-1)

        # Computing answer
        logits = self._compute_logits(x=stacked_features, training=training)

        return logits, {
            'raw_predictions': {key: tf.nn.softmax(value) for key, value in logits.items()}
        }


class M_MArgBERT(tf.keras.Model):

    def __init__(self, label_list, bert_config, preloaded_model_name, max_frame_length,
                 max_frame_features_length, dropout_text=0.1, answer_units=100, answer_dropout=0.1,
                 audio_units=128, audio_l2=0.5, dropout_audio=0.2, audio_layers=None,
                 is_bert_trainable=False, data_mode='text_only', use_audio_features=True,
                 ablation_study=None):
        super(M_MArgBERT, self).__init__()
        self.max_frame_length = max_frame_length
        self.max_frame_features_length = max_frame_features_length
        self.data_mode = data_mode
        self.label_list = label_list
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        if self.data_mode != 'text_audio':
            self.ablation_study = None

        # Text
        if self.data_mode != 'audio_only':
            self.bert = TFBertModel.from_pretrained(pretrained_model_name_or_path=preloaded_model_name,
                                                    config=bert_config, name='bert')
            if not is_bert_trainable:
                for layer in self.bert.layers:
                    layer.trainable = False

            self.text_dropout = tf.keras.layers.Dropout(rate=dropout_text)

        # Audio
        if self.data_mode != 'text_only':
            self.audio_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=audio_units,
                                                                                 return_sequences=True,
                                                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                                                     l2=audio_l2)))
            self.audio_dropout = tf.keras.layers.Dropout(rate=dropout_audio)

            self.audio_cnn_blocks = []
            self.audio_layers = audio_layers if audio_layers is not None else []
            for audio_layer in self.audio_layers:
                audio_cnn = tf.keras.layers.Conv2D(filters=audio_layer['filters'],
                                                   kernel_size=audio_layer['kernel_size'],
                                                   strides=audio_layer['kernel_strides'],
                                                   padding='valid')
                self.audio_cnn_blocks.append(audio_cnn)
                self.audio_cnn_blocks.append(tf.keras.layers.BatchNormalization())
                self.audio_cnn_blocks.append(tf.keras.layers.Activation('relu'))
                self.audio_cnn_blocks.append(tf.keras.layers.Dropout(rate=dropout_audio))
                self.audio_cnn_blocks.append(tf.keras.layers.MaxPool2D(pool_size=audio_layer['pool_size'],
                                                                       strides=audio_layer['pool_strides']))

        # Answer
        self.answer_blocks = [
            tf.keras.layers.Dense(units=answer_units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(rate=answer_dropout)
        ]

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def _convert_text(self, text_ids, text_attention_mask, training=False):
        text_emb = self.bert({
            'input_ids': text_ids,
            'attention_mask': text_attention_mask
        }, training=training)[1]
        text_emb = self.text_dropout(text_emb, training=training)

        return text_emb

    def _convert_audio(self, audio_features, training=False):
        # LSTM embedding
        audio_emb = self.audio_lstm(audio_features, training=training)
        audio_emb = tf.reshape(audio_emb, [audio_emb.shape[0], -1])
        audio_emb = self.audio_dropout(audio_emb, training=training)

        # CNN embedding
        if len(self.audio_cnn_blocks):
            audio_cnn = tf.expand_dims(audio_features, axis=-1)
            for block in self.audio_cnn_blocks:
                audio_cnn = block(audio_cnn, training=training)

            audio_cnn = tf.reshape(audio_cnn, [audio_cnn.shape[0], -1])

            audio_emb = tf.concat((audio_emb, audio_cnn), axis=-1)

        return audio_emb

    def _compute_logits(self, features, training=False):
        answer = features
        for block in self.answer_blocks:
            answer = block(answer, training=training)

        logits = {key: block(answer, training=training) for key, block in self.final_block.items()}
        return logits

    def call(self, inputs, training=False, **kwargs):
        stacked_features = None

        # Text
        if self.data_mode != 'audio_only':
            text_a_emb = self._convert_text(text_ids=inputs['text_a_ids'],
                                            text_attention_mask=inputs['text_a_attention_mask'],
                                            training=training)

            text_b_emb = self._convert_text(text_ids=inputs['text_b_ids'],
                                            text_attention_mask=inputs['text_b_attention_mask'],
                                            training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'text':
                text_a_emb = tf.zeros_like(text_a_emb)
                text_b_emb = tf.zeros_like(text_b_emb)

            stacked_features = tf.concat((text_a_emb, text_b_emb), axis=-1)

        # Audio
        if self.data_mode != 'text_only':
            if self.use_audio_features:
                audio_a_mfccs = inputs['audio_a_mfccs']
                audio_a_mfccs = tf.reshape(audio_a_mfccs, [audio_a_mfccs.shape[0], -1, self.max_frame_features_length])
                audio_a_emb = self._convert_audio(audio_features=audio_a_mfccs,
                                                               training=training)

                audio_b_mfccs = inputs['audio_b_mfccs']
                audio_b_mfccs = tf.reshape(audio_b_mfccs, [audio_b_mfccs.shape[0], -1, self.max_frame_features_length])
                audio_b_emb = self._convert_audio(audio_features=audio_b_mfccs,
                                                               training=training)
            else:
                audio_a_data = inputs['audio_a_data']
                audio_a_emb = self._convert_audio(audio_features=audio_a_data[:, None, :], training=training)

                audio_b_data = inputs['audio_b_data']
                audio_b_emb = self._convert_audio(audio_features=audio_b_data[:, None, :], training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'audio':
                audio_a_emb = tf.zeros_like(audio_a_emb)
                audio_b_emb = tf.zeros_like(audio_b_emb)

            if self.data_mode == 'audio_only':
                stacked_features = tf.concat((audio_a_emb, audio_b_emb), axis=-1)
            else:
                stacked_features = tf.concat((stacked_features, audio_a_emb, audio_b_emb), axis=-1)

        logits = self._compute_logits(features=stacked_features, training=training)

        return logits, {
            'raw_predictions': {key: tf.nn.softmax(value) for key, value in logits.items()}
        }


class M_UsElecLSTM(tf.keras.Model):

    def __init__(self, label_list, embedding_dimension, vocab_size,
                 max_speech_length, max_frame_length, max_frame_features_length,
                 lstm_weights, answer_weights, use_bidirectional=False,
                 data_mode='text_only', l2_regularization=0.,
                 dropout_rate=0.2, embedding_matrix=None, use_audio_features=True,
                 ablation_study=None):
        super(M_UsElecLSTM, self).__init__()
        self.text_length = max_speech_length
        self.max_frame_length = max_frame_length
        self.max_frame_features_length = max_frame_features_length
        self.vocab_size = vocab_size
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension
        self.label_list = label_list
        self.data_mode = data_mode
        self.use_bidirectional = use_bidirectional
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        if self.data_mode != 'text_audio':
            self.ablation_study = None

        if self.data_mode != 'audio_only':
            self.input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                             output_dim=embedding_dimension,
                                                             input_length=self.text_length,
                                                             weights=[embedding_matrix] if embedding_matrix is not None
                                                             else embedding_matrix,
                                                             mask_zero=True,
                                                             name='input_embedding')

            self.text_lstm_blocks = []
            for weight_idx, weight in enumerate(self.lstm_weights):
                lstm_block = tf.keras.layers.LSTM(weight,
                                                  return_sequences=False
                                                  if weight_idx == len(self.lstm_weights) - 1 else True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                      self.l2_regularization))

                if self.use_bidirectional:
                    lstm_block = tf.keras.layers.Bidirectional(lstm_block)

                self.text_lstm_blocks.append(lstm_block)
                self.text_lstm_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        if self.data_mode != 'text_only':
            self.audio_lstm_blocks = []
            for weight_idx, weight in enumerate(self.lstm_weights):
                lstm_block = tf.keras.layers.LSTM(weight,
                                                  return_sequences=False
                                                  if weight_idx == len(self.lstm_weights) - 1 else True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                      self.l2_regularization))
                if self.use_bidirectional:
                    lstm_block = tf.keras.layers.Bidirectional(lstm_block)

                self.audio_lstm_blocks.append(lstm_block)
                self.audio_lstm_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.answer_blocks = []
        for weight in answer_weights:
            self.answer_blocks.append(tf.keras.layers.Dense(units=weight,
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_regularizer=tf.keras.regularizers.l2(
                                                                l2_regularization)))
            self.answer_blocks.append(tf.keras.layers.Dropout(rate=dropout_rate))

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              kernel_regularizer=tf.keras.regularizers.l2(
                                                                  l2_regularization),
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def _convert_text(self, x, training=False):
        for block in self.text_lstm_blocks:
            x = block(x, training=training)
        return x

    def _convert_audio(self, x, training=False):
        for block in self.audio_lstm_blocks:
            x = block(x, training=training)
        return x

    def _compute_logits(self, x, training=False):
        for block in self.answer_blocks:
            x = block(x, training=training)
        return x

    def call(self, inputs, training=False, **kwargs):
        stacked_features = None

        # Converting speech text (if any)
        if self.data_mode != 'audio_only':
            # [batch_size, max_speech_length]
            speech_ids = inputs['speech_ids']

            # [batch_size, max_speech_length, emb_dim]
            speech_emb = self.input_embedding(speech_ids)

            # [batch_size, hidden_size]
            speech_emb = self._convert_text(x=speech_emb, training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'text':
                speech_emb = tf.zeros_like(speech_emb)

            # [batch_size, hidden_size * 2]
            stacked_features = speech_emb

        # Converting audio features (if any)
        if self.data_mode != 'text_only':
            # [batch_size, max_frame_length, mfccs]
            if self.use_audio_features:
                speech_mfccs = inputs['speech_mfccs']
                speech_mfccs = tf.reshape(speech_mfccs, [-1, self.max_frame_length, self.max_frame_features_length])

                # [batch_size, audio_hidden_size]
                audio_emb = self._convert_audio(x=speech_mfccs, training=training)
            else:
                speech_audio = inputs['speech_audio']

                # [batch_size, audio_hidden_size]
                audio_emb = self._convert_audio(x=speech_audio[:, None, :], training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'audio':
                audio_emb = tf.zeros_like(audio_emb)

            # [batch_size, hidden_size + audio_hidden_size] <-- audio_only
            # [batch_size, hidden_size * 2 + audio_hidden_size] <-- text_audio
            if stacked_features is None:
                stacked_features = audio_emb
            else:
                stacked_features = tf.concat((stacked_features, audio_emb), axis=-1)

        # Computing answer
        answer = self._compute_logits(x=stacked_features, training=training)
        logits = {key: block(answer, training=training) for key, block in self.final_block.items()}

        return logits, {
            'raw_predictions': {key: tf.nn.softmax(value) for key, value in logits.items()}
        }


class M_UsElecBERT(tf.keras.Model):

    def __init__(self, label_list, bert_config, preloaded_model_name, max_frame_length,
                 max_frame_features_length, dropout_text=0.1, answer_units=100, answer_dropout=0.1,
                 audio_units=128, audio_l2=0.5, dropout_audio=0.2, audio_layers=None,
                 is_bert_trainable=False, data_mode='text_only', use_audio_features=True,
                 ablation_study=None):
        super(M_UsElecBERT, self).__init__()
        self.max_frame_length = max_frame_length
        self.max_frame_features_length = max_frame_features_length
        self.data_mode = data_mode
        self.label_list = label_list
        self.use_audio_features = use_audio_features
        self.ablation_study = ablation_study

        if self.data_mode != 'text_audio':
            self.ablation_study = None

        # Text
        if self.data_mode != 'audio_only':
            self.bert = TFBertModel.from_pretrained(pretrained_model_name_or_path=preloaded_model_name,
                                                    config=bert_config, name='bert')
            if not is_bert_trainable:
                for layer in self.bert.layers:
                    layer.trainable = False

            self.text_dropout = tf.keras.layers.Dropout(rate=dropout_text)

        # Audio
        if self.data_mode != 'text_only':
            self.audio_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=audio_units,
                                                                                 return_sequences=True,
                                                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                                                     l2=audio_l2)))
            self.audio_dropout = tf.keras.layers.Dropout(rate=dropout_audio)

            self.audio_cnn_blocks = []
            self.audio_layers = audio_layers if audio_layers is not None else []
            for audio_layer in self.audio_layers:
                audio_cnn = tf.keras.layers.Conv2D(filters=audio_layer['filters'],
                                                   kernel_size=audio_layer['kernel_size'],
                                                   strides=audio_layer['kernel_strides'],
                                                   padding='valid')
                self.audio_cnn_blocks.append(audio_cnn)
                self.audio_cnn_blocks.append(tf.keras.layers.BatchNormalization())
                self.audio_cnn_blocks.append(tf.keras.layers.Activation('relu'))
                self.audio_cnn_blocks.append(tf.keras.layers.Dropout(rate=dropout_audio))
                self.audio_cnn_blocks.append(tf.keras.layers.MaxPool2D(pool_size=audio_layer['pool_size'],
                                                                       strides=audio_layer['pool_strides']))

        # Answer
        self.answer_blocks = [
            tf.keras.layers.Dense(units=answer_units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(rate=answer_dropout)
        ]

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def _convert_text(self, text_ids, text_attention_mask, training=False):
        text_emb = self.bert({
            'input_ids': text_ids,
            'attention_mask': text_attention_mask
        }, training=training)[1]
        text_emb = self.text_dropout(text_emb, training=training)

        return text_emb

    def _convert_audio(self, audio_features, training=False):
        # LSTM embedding
        audio_emb = self.audio_lstm(audio_features, training=training)
        audio_emb = tf.reshape(audio_emb, [audio_emb.shape[0], -1])
        audio_emb = self.audio_dropout(audio_emb, training=training)

        # CNN embedding
        if len(self.audio_cnn_blocks):
            audio_cnn = tf.expand_dims(audio_features, axis=-1)
            for block in self.audio_cnn_blocks:
                audio_cnn = block(audio_cnn, training=training)

            audio_cnn = tf.reshape(audio_cnn, [audio_cnn.shape[0], -1])

            audio_emb = tf.concat((audio_emb, audio_cnn), axis=-1)

        return audio_emb

    def _compute_logits(self, features, training=False):
        answer = features
        for block in self.answer_blocks:
            answer = block(answer, training=training)

        logits = {key: block(answer, training=training) for key, block in self.final_block.items()}
        return logits

    def call(self, inputs, training=False, **kwargs):
        stacked_features = None

        # Converting speech text (if any)
        if self.data_mode != 'audio_only':
            speech_emb = self._convert_text(text_ids=inputs['speech_ids'],
                                            text_attention_mask=inputs['speech_attention_mask'],
                                            training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'text':
                speech_emb = tf.zeros_like(speech_emb)

            stacked_features = speech_emb

        # Converting audio features (if any)
        if self.data_mode != 'text_only':
            if self.use_audio_features:
                # [batch_size, max_frame_length, mfccs]
                speech_mfccs = inputs['speech_mfccs']
                speech_mfccs = tf.reshape(speech_mfccs, [speech_mfccs.shape[0], -1, self.max_frame_features_length])

                audio_emb = self._convert_audio(audio_features=speech_mfccs, training=training)
            else:
                speech_audio = inputs['speech_audio']

                # [batch_size, audio_hidden_size]
                audio_emb = self._convert_audio(audio_features=speech_audio[:, None, :], training=training)

            if self.data_mode == 'text_audio' and self.ablation_study == 'audio':
                audio_emb = tf.zeros_like(audio_emb)

            if stacked_features is None:
                stacked_features = audio_emb
            else:
                stacked_features = tf.concat((stacked_features, audio_emb), axis=-1)

        logits = self._compute_logits(features=stacked_features, training=training)

        return logits, {
            'raw_predictions': {key: tf.nn.softmax(value) for key, value in logits.items()}
        }
