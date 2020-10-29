import tensorflow as tf
from numpy.testing import assert_array_equal

from crnn.layers import ConvToRec


def test_call__output_shape_is_correct():
    conv_to_rec = ConvToRec()
    inputs = tf.random.uniform([32, 9, 12, 128])

    outputs = conv_to_rec(inputs)

    assert_array_equal(outputs.shape, [32, 12, 9*128])


def test_call__output_values_are_correct():
    conv_to_rec = ConvToRec()
    inputs = tf.random.uniform([32, 9, 12, 128])

    outputs = conv_to_rec(inputs)

    for i in range(inputs.shape[2]):
        assert_array_equal(outputs[:, i], tf.reshape(inputs[:, :, i, :], [inputs.shape[0], -1]))
