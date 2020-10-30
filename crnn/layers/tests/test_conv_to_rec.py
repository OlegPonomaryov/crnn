"""Tests for the ConvToRec layer"""
import numpy as np

from numpy.testing import assert_array_equal

from crnn.layers import ConvToRec


def test_call__output_shape_is_correct():
    samples, rows, columns, channels = 32, 8, 24, 128
    inputs = np.random.uniform(size=[samples, rows, columns, channels]).astype(np.float32)
    conv_to_rec = ConvToRec()

    outputs = conv_to_rec(inputs)

    assert_array_equal(outputs.shape, [samples, columns, rows*channels])


def test_call__output_values_are_correct():
    samples, rows, columns, channels = 32, 8, 24, 128
    inputs = np.random.uniform(size=[samples, rows, columns, channels]).astype(np.float32)
    conv_to_rec = ConvToRec()

    outputs = conv_to_rec(inputs)

    for i in range(columns):
        column = inputs[:, :, i, :]
        assert_array_equal(outputs[:, i], np.reshape(column, [samples, -1]))
