
import os
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.models import BaseNetwork, ClassificationNetwork, GenerativeNetwork
from deasy_learning_generic.implementations.labels import ClassificationLabel, RegressionLabel, GenerativeLabel
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.printing_utils import prettify_statistics
from deasy_learning_generic.utility.metric_utils import compute_metrics_error
from deasy_learning_generic.utility.python_utils import merge
from deasy_learning_generic.utility.pickle_utils import save_pickle, load_pickle
from collections import OrderedDict
import gc
from tensorflow.keras import backend as k
from deasy_learning_tf.utility.tensorflow_utils import assert_no_variable_creations, catch_and_raise_created_variables

logger = Logger.get_logger(__name__)



class TFNetwork(BaseNetwork):

    # Saving/Weights

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save(self, filepath, overwrite=True):
        if filepath.endswith('.h5'):
            base_path = filepath.split('.h5')[0]
        else:
            base_path = filepath

        # Model weights
        weights_path = base_path + '.h5'
        self.model.save_weights(weights_path)

        # Model state
        network_state = self.get_state()
        if network_state is not None:
            save_pickle(filepath=base_path + '.pickle', data=network_state)

    def load(self, filepath, **kwargs):
        if filepath.endswith('.h5'):
            base_path = filepath.split('.h5')[0]
            weights_path = base_path + '.h5'
        else:
            base_path = filepath
            weights_path = filepath + '.h5'

        # Model weights
        self.model.load_weights(filepath=weights_path, **kwargs)

        # Model state
        state_path = base_path + '.pickle'
        if os.path.isfile(state_path):
            model_state = load_pickle(filepath=state_path)
            self.set_state(model_state=model_state)

    # Routine

    def prepare_for_loading(self, data):
        self.predict(data=data, is_toy=True)
        self.initial_weights = [layer.get_weights() for layer in self.model.layers]

    def check_after_loading(self):
        # Correct loading check (inherently makes sure that restore ops are run)
        for layer, initial in zip(self.model.layers, self.initial_weights):
            weights = layer.get_weights()
            if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                Logger.get_logger(__name__).info('Checkpoint contained no weights for layer {}!'.format(layer.name))

        # Flush some memory
        del self.initial_weights

    # Training/Inference

    def predict(self, data, callbacks=None, repetitions=1, suffix=DataSplit.TEST, metrics=None, is_toy=False):

        avg_predictions = []
        avg_metrics = {}

        if metrics is not None:
            data_labels = data.get_labels()
        else:
            data_labels = None

        for rep in range(repetitions):
            rep_predictions = self._predict(data=data, callbacks=callbacks, suffix=suffix, is_toy=is_toy)
            avg_predictions.append(rep_predictions)

            if metrics is not None and data_labels is not None:
                ground_truth = self.parse_labels(data_labels)
                metric_results = compute_metrics_error(predicted_values=rep_predictions,
                                                       true_values=ground_truth,
                                                       prefix=suffix,
                                                       metrics=metrics)
                for key, value in metric_results.items():
                    avg_metrics.setdefault(key, []).append(value)

        if metrics is not None:
            avg_metrics = {key: np.mean(value, axis=0) for key, value in avg_metrics.items()}

        return avg_predictions, avg_metrics

    def _predict(self, data, callbacks=None, suffix=DataSplit.TEST, is_toy=False):
        callbacks = callbacks or []

        total_predictions = {}

        data_iterator = data.get_data()
        data_steps = data.steps if not is_toy else 1

        for callback in callbacks:
            if callback.model is None:
                callback.set_model(model=self)
            callback.on_prediction_begin(logs={'suffix': suffix})

        for batch_idx in tqdm(range(data_steps), leave=True, position=0):

            for callback in callbacks:
                callback.on_batch_prediction_begin(batch=batch_idx, logs={'suffix': suffix})

            batch = next(data_iterator)
            if type(batch) in [tuple, list]:
                batch = batch[0]
            preds, model_additional_info = self.batch_predict(x=batch)
            preds = {key: self.parse_predictions(value.numpy(), model_additional_info) for key, value in preds.items()}
            for key, value in preds.items():
                total_predictions.setdefault(key, []).extend(value)

            for callback in callbacks:
                callback.on_batch_prediction_end(batch=batch_idx, logs={'predictions': preds,
                                                                        'model_additional_info': model_additional_info,
                                                                        'suffix': suffix})

        for callback in callbacks:
            callback.on_prediction_end(logs={'suffix': suffix})

        return total_predictions

    def _evaluate_and_predict(self, data, callbacks=None, suffix=DataSplit.VAL):
        total_loss = {}

        callbacks = callbacks or []

        total_preds = {}

        data_iterator = data.get_data()
        data_steps = data.steps

        for callback in callbacks:
            if callback.model is None:
                callback.set_model(model=self)
            callback.on_prediction_begin(logs={'suffix': suffix})

        for batch_idx in tqdm(range(data_steps), leave=True, position=0):

            for callback in callbacks:
                callback.on_batch_prediction_begin(batch=batch_idx, logs={'suffix': suffix})

            batch = next(data_iterator)
            batch_additional_info = self._get_additional_info()
            loss, loss_info, preds, model_additional_info = self.loss_op(x=batch[0], targets=batch[1],
                                                                         training=False,
                                                                         state='evaluation',
                                                                         return_predictions=True,
                                                                         additional_info=batch_additional_info)

            batch_info = {'val_{}'.format(key): item for key, item in loss_info.items()}
            batch_info['val_loss'] = loss

            batch_info = {key: item.numpy() for key, item in batch_info.items()}

            for key, item in batch_info.items():
                if key not in total_loss:
                    total_loss[key] = item
                else:
                    total_loss[key] += item

            preds = {key: self.parse_predictions(value.numpy(), model_additional_info) for key, value in preds.items()}
            for key, value in preds.items():
                total_preds.setdefault(key, []).extend(value)

            for callback in callbacks:
                callback.on_batch_prediction_end(batch=batch_idx, logs={'predictions': preds,
                                                                        'model_additional_info': model_additional_info,
                                                                        'suffix': suffix})

        for callback in callbacks:
            callback.on_prediction_end(logs={'suffix': suffix})

        total_loss = {key: item / data_steps for key, item in total_loss.items()}

        return total_loss, total_preds

    def evaluate(self, data, callbacks=None, repetitions=1, metrics=None):

        avg_val_info = {}
        avg_val_metrics = {}

        if metrics is not None:
            data_labels = data.get_labels()
        else:
            data_labels = None

        for rep in range(repetitions):
            val_info, val_preds = self._evaluate_and_predict(data=data, callbacks=callbacks, suffix=DataSplit.VAL)
            for key, value in val_info.items():
                avg_val_info.setdefault(key, []).append(value)

            if metrics is not None and data_labels is not None:
                val_y = self.parse_labels(data_labels)
                all_val_metrics = compute_metrics_error(predicted_values=val_preds,
                                                        true_values=val_y,
                                                        prefix=DataSplit.VAL,
                                                        metrics=metrics,
                                                        )
                for key, value in all_val_metrics.items():
                    avg_val_metrics.setdefault(key, []).append(value)

        avg_val_info = {key: np.mean(value, axis=0) for key, value in avg_val_info.items()}
        if metrics is not None:
            avg_val_metrics = {key: np.mean(value, axis=0) for key, value in avg_val_metrics.items()}

        return avg_val_info, avg_val_metrics

    def fit(self, train_data, epochs=1, verbose=1,
            callbacks=None, validation_data=None, step_checkpoint=None,
            metrics=None, inference_repetitions=1):

        shuffled_train_iterator = train_data.get_training_iterator()
        train_steps = train_data.steps

        callbacks = callbacks or []
        for callback in callbacks:
            callback.set_model(model=self)
            callback.on_train_begin(logs={'epochs': epochs,
                                          'steps_per_epoch': train_steps})

        if verbose:
            logger.info('Start Training!')

            if train_steps is not None:
                logger.info('Total batches: {}'.format(train_steps))

        if step_checkpoint is not None:
            if type(step_checkpoint) == float:
                step_checkpoint = int(train_steps * step_checkpoint)
                logger.info('Converting percentage step checkpoint to: {}'.format(step_checkpoint))
            else:
                if step_checkpoint > train_steps:
                    step_checkpoint = int(train_steps * 0.1)
                    logger.info('Setting step checkpoint to: {}'.format(step_checkpoint))

        metrics.update_metrics_with_model_info(model=self)

        # Training
        for epoch in range(epochs):

            if hasattr(self.model, 'stop_training') and self.model.stop_training:
                break

            for callback in callbacks:
                callback.on_epoch_begin(epoch=epoch, logs={'epochs': epochs})

            train_loss = {}
            batch_idx = 0

            # Run epoch
            pbar = tqdm(total=train_steps, leave=True, position=0)
            while batch_idx < train_steps:

                for callback in callbacks:
                    callback.on_batch_begin(batch=batch_idx, logs=None)

                batch_additional_info = self._get_additional_info()

                if batch_idx > 0:
                    with assert_no_variable_creations():
                        with catch_and_raise_created_variables():
                            batch_info, model_additional_info = self.batch_fit(*next(shuffled_train_iterator),
                                                                               batch_additional_info)
                else:
                    batch_info, model_additional_info = self.batch_fit(*next(shuffled_train_iterator),
                                                                       batch_additional_info)

                batch_info = {key: item.numpy() for key, item in batch_info.items()}

                for callback in callbacks:
                    callback.on_batch_end(batch=batch_idx, logs=batch_info)

                # Update any internal network state
                self._update_internal_state(model_additional_info)

                for key, item in batch_info.items():
                    if key in train_loss:
                        train_loss[key] += item
                    else:
                        train_loss[key] = item

                batch_idx += 1
                pbar.update(1)

            pbar.close()

            train_loss = {key: item / train_steps for key, item in train_loss.items()}
            train_loss_str = {key: float('{:.2f}'.format(value)) for key, value in train_loss.items()}

            val_info = None

            # Compute metrics at the end of each epoch
            callback_additional_args = {}

            if validation_data is not None:
                val_info, all_val_metrics = self.evaluate(data=validation_data,
                                                          callbacks=callbacks,
                                                          metrics=metrics,
                                                          repetitions=inference_repetitions)

                val_info_str = {key: value for key, value in val_info.items()}

                if metrics is not None:
                    val_metrics_str_result = {key: value for key, value in
                                              all_val_metrics.items()}

                    merged_statistics = merge(train_loss_str, val_info_str)
                    merged_statistics = merge(merged_statistics, val_metrics_str_result)
                    merged_statistics = merge(merged_statistics, {'epoch': epoch + 1})

                    logger.info('\n{}'.format(prettify_statistics(merged_statistics)))

                    callback_additional_args = all_val_metrics
                else:
                    if verbose:
                        merged_statistics = merge(train_loss_str, val_info_str)
                        merged_statistics = merge(merged_statistics, {'epoch': epoch + 1})
                        logger.info(prettify_statistics(merged_statistics))
            else:
                merged_statistics = merge(train_loss_str, {'epoch': epoch + 1})
                logger.info(prettify_statistics(merged_statistics))

            for callback in callbacks:
                callback_args = merge(train_loss, val_info)
                callback_args = merge(callback_args,
                                      callback_additional_args,
                                      overwrite_conflict=False)
                callback.on_epoch_end(epoch=epoch, logs=callback_args)

            # Garbage collect
            gc.collect()
            k.clear_session()

        for callback in callbacks:
            callback.on_train_end(logs=None)

    @tf.function
    def batch_fit(self, x, y, additional_info=None):
        loss, loss_info, model_additional_info, grads = self.train_op(x, y, additional_info=additional_info)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        train_loss_info = {'train_{}'.format(key): item for key, item in loss_info.items()}
        train_loss_info['train_loss'] = loss
        return train_loss_info, model_additional_info

    @tf.function
    def distributed_batch_fit(self, inputs, strategy):
        train_loss_info = strategy.run(self.batch_fit, args=inputs)
        train_loss_info = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, item, axis=None)
                           for key, item in train_loss_info.items()}
        return train_loss_info

    @tf.function
    def batch_predict(self, x):
        additional_info = self._get_additional_info()
        predictions, model_additional_info = self.model(x,
                                                        state='prediction',
                                                        training=False,
                                                        additional_info=additional_info)
        return predictions, model_additional_info

    @tf.function
    def distributed_batch_predict(self, inputs, strategy):
        predictions = strategy.run(self.batch_predict, args=inputs)
        return predictions

    @tf.function
    def distributed_batch_evaluate(self, inputs, strategy):
        val_loss_info = strategy.run(self.batch_evaluate, args=inputs)
        val_loss_info = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, item, axis=None)
                         for key, item in val_loss_info.items()}
        return val_loss_info

    @tf.function
    def batch_evaluate(self, x, y, additional_info=None):
        loss, loss_info = self.loss_op(x, y, training=False, state='evaluation', additional_info=additional_info)
        val_loss_info = {'val_{}'.format(key): item for key, item in loss_info.items()}
        val_loss_info['val_loss'] = loss
        return val_loss_info

    @tf.function
    def batch_evaluate_and_predict(self, x, y, additional_info=None):
        loss, loss_info, predictions, model_additional_info = self.loss_op(x, y, training=False,
                                                                           state='evaluation',
                                                                           additional_info=additional_info,
                                                                           return_predictions=True)
        val_loss_info = {'val_{}'.format(key): item for key, item in loss_info.items()}
        val_loss_info['val_loss'] = loss
        return val_loss_info, predictions, model_additional_info

    # Model definition

    @tf.function
    def train_op(self, x, y, additional_info):
        with tf.GradientTape() as tape:
            loss, loss_info, model_additional_info = self.loss_op(x, y, training=True, additional_info=additional_info)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, loss_info, model_additional_info, grads


class TFClassificationNetwork(TFNetwork, ClassificationNetwork):

    def _classification_ce(self, targets, logits, label_name, reduce=True):
        label_class_weights = self.class_weights[label_name]

        if len(logits.shape) > 2:
            logits = tf.reshape(logits, [-1, logits.shape[-1]])
            targets = tf.reshape(targets, [-1])

        if len(logits.shape) == len(targets.shape):
            targets = tf.reshape(targets, [-1])

        # [batch_size,]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                       logits=logits)

        if self.weight_predictions:
            label_weights = tf.ones(shape=targets.shape[0], dtype=logits.dtype)
            if len(targets.shape) > 1:
                key_target_classes = tf.argmax(targets, axis=-1)
            else:
                key_target_classes = targets
            for cls, weight in label_class_weights.items():
                to_fill = tf.cast(tf.fill(label_weights.shape, value=weight), logits.dtype)
                label_weights = tf.where(key_target_classes == cls, to_fill, label_weights)

            cross_entropy *= label_weights

        if reduce:
            cross_entropy = tf.reduce_mean(cross_entropy)

        return cross_entropy

    def _regression_loss(self, targets, logits):
        difference = tf.math.squared_difference(logits, targets)
        return tf.reduce_mean(difference)

    def _compute_losses(self, targets, logits, label_list, reduce=True):
        total_loss = None
        loss_info = {}
        for label_idx, label in enumerate(label_list):
            label_targets = targets[label.name]
            label_logits = logits[label.name]

            if isinstance(label, ClassificationLabel):
                loss = self._classification_ce(targets=label_targets,
                                               logits=label_logits,
                                               label_name=label.name,
                                               reduce=reduce)
            elif isinstance(label, RegressionLabel):
                loss = self._regression_loss(targets=label_targets,
                                             logits=label_logits)
            else:
                raise RuntimeError("Invalid label type -> {}".format(type(label)))

            loss_info.setdefault(label.name, loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_info

    def parse_predictions(self, raw_predictions, model_additional_info):
        if type(raw_predictions) in [dict, OrderedDict]:
            return {key: np.argmax(value, axis=-1) for key, value in raw_predictions.items()}

        return np.argmax(raw_predictions, axis=-1)


class TFGenerativeNetwork(TFNetwork, GenerativeNetwork):

    def _compute_losses(self, targets, logits, label_list):
        total_loss = None
        loss_info = {}
        for label_idx, label in enumerate(label_list):
            label_targets = targets[label.name]
            label_logits = logits[label.name]

            if isinstance(label, GenerativeLabel):
                loss = self._classification_ce(targets=label_targets,
                                               logits=label_logits)
            else:
                raise RuntimeError("Invalid label type -> {}".format(type(label)))

            loss_info.setdefault(label.name, loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_info

    def _classification_ce(self, targets, logits):

        if len(logits.shape) > 2:
            logits = tf.reshape(logits, [-1, logits.shape[-1]])
            targets = tf.reshape(targets, [-1])

        if len(logits.shape) == len(targets.shape):
            targets = tf.reshape(targets, [-1])

        # [batch_size,]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                       logits=logits)

        return tf.reduce_mean(cross_entropy)

    def generate(self, x):
        generated = self.model.generate(input_ids=x['input_ids'],
                                        attention_mask=x['attention_mask'],
                                        max_length=self.max_generation_length)
        generated = {self.label_list[0].name: generated}

        return generated
