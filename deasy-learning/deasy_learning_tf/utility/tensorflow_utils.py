
import numpy as np
import tensorflow as tf
from contextlib import contextmanager

from deasy_learning_generic.utility.python_utils import merge


def stable_norm(x, axis=None):
    """
    tf.norm is not numerically stable when computing gradient (wtf).
    Link: https://datascience.stackexchange.com/questions/80898/tensorflow-gradient-returns-nan-or-inf
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + 1.0e-08)


def add_gradient_noise(t, stddev=1e-3, name="add_gradient_noise"):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """

    with tf.name_scope(name) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random.normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    Positional encoding adopted in the Transformer implementation
    as described in https://www.tensorflow.org/tutorials/text/transformer
    """

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def sparse_softmax(tensor, mask):
    """
    Sparse softmax that ignores padding values (0s).
    """

    indexes = tf.where(mask)
    sparse_tensor = tf.SparseTensor(indexes, tf.gather_nd(tensor, indexes), tensor.get_shape())
    sparse_probs = tf.sparse.softmax(sparse_tensor)
    dense_probs = tf.sparse.to_dense(sparse_probs)

    return dense_probs


def decode_record(record, name_to_features):
    """
    TPU does not support int64
    """

    example = tf.io.parse_single_example(record, name_to_features)

    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def load_single_dataset(filepath, name_to_features):
    data = tf.data.TFRecordDataset(filepath)
    data = data.map(lambda record: decode_record(record, name_to_features))

    return data


def create_dataset(filepath, batch_size, name_to_features, selector, is_training=True,
                   input_pipeline_context=None, shuffle_amount=10000, prefetch_amount=1024,
                   reshuffle_each_iteration=True, sampling=False, sampler=None):
    dataset = load_single_dataset(filepath=filepath, name_to_features=name_to_features)

    # Dataset is sharded by the number of hosts (num_input_pipelines == num_hosts)
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)

    dataset = dataset.map(selector, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_amount, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.repeat()
        if sampling:
            dataset = dataset.map(lambda x, y: sampler.sampling((x, y)),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(prefetch_amount)
    return dataset


def get_dataset_fn(filepath, batch_size, name_to_features, selector, is_training=True,
                   shuffle_amount=10000, prefetch_amount=1024,
                   reshuffle_each_iteration=True,
                   sampling=False, sampler=None):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset"""
        bs = ctx.get_per_replica_batch_size(batch_size) if ctx else batch_size
        dataset = create_dataset(filepath=filepath, batch_size=bs,
                                 name_to_features=name_to_features,
                                 selector=selector,
                                 is_training=is_training,
                                 input_pipeline_context=ctx,
                                 shuffle_amount=shuffle_amount,
                                 reshuffle_each_iteration=reshuffle_each_iteration,
                                 prefetch_amount=prefetch_amount,
                                 sampling=sampling,
                                 sampler=sampler)
        return dataset

    return _dataset_fn


def retrieve_numpy_labels(data_fn, steps):
    numpy_data = list(data_fn().map(lambda x, y: y).take(steps).as_numpy_iterator())
    if type(numpy_data[0]) == dict:
        numpy_data = {key: np.concatenate([item[key] for item in numpy_data]) for key in numpy_data[0].keys()}
    else:
        numpy_data = np.concatenate([item for item in data_fn().map(lambda x, y: y).take(steps)])

    return numpy_data


def transform_to_pairwise(data_function, other_data_function=None, is_callable=True):
    if other_data_function is None:
        d1 = data_function
        d2 = data_function
    else:
        d1 = data_function
        d2 = other_data_function

    if is_callable:
        dataset = tf.data.Dataset.zip((d1(), d2()))
    else:
        dataset = tf.data.Dataset.zip((d1, d2))
    dataset = dataset.map(lambda a, b: (merge({'left_{}'.format(key): value for key, value in a[0].items()},
                                              {'right_{}'.format(key): value for key, value in b[0].items()}),
                                        tf.where(tf.argmax(a[1], axis=1) == tf.argmax(b[1], axis=1),
                                                 1,
                                                 -1)
                                        ))

    return dataset


@contextmanager
def assert_no_variable_creations():
    """Assert no variables are created in this context manager scope."""

    def invalid_variable_creator(next_creator, **kwargs):
        raise ValueError(
            "Attempted to create a new variable instead of reusing an existing one. Args: {}".format(kwargs))

    with tf.variable_creator_scope(invalid_variable_creator):
        yield


@contextmanager
def catch_and_raise_created_variables():
    """Raise all variables created within this context manager scope (if any)."""
    created_vars = []

    def variable_catcher(next_creator, **kwargs):
        var = next_creator(**kwargs)
        created_vars.append(var)
        return var

    with tf.variable_creator_scope(variable_catcher):
        yield
    if created_vars:
        raise ValueError("Created vars:", created_vars)
