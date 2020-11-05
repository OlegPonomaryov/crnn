"""Tests for the CTC accuracy"""
import tensorflow as tf

import pytest
from numpy.testing import assert_array_equal

from crnn.metrics import ctc_accuracy


def test_ctc_accuracy__padded_and_not_padded_labels():
    classes_count = 10
    pad_const = -1
    y_true = [[4, 2]]
    y_true_padded = [[4, 2, pad_const, pad_const]]
    y_pred = tf.one_hot([[classes_count, 4, classes_count, 2, 2]], classes_count + 1)

    loss_without_padding = ctc_accuracy(y_true, y_pred)
    loss_with_padding = ctc_accuracy(y_true_padded, y_pred, y_true_padding_const=pad_const)

    # Padding of the labels shouldn't affect the accuracy value
    assert_array_equal(loss_without_padding, loss_with_padding)


# pad_const argument is used to check that CTC accuracy works if y_true padding const both equal and not equal to the
# ctc_decode()'s unknown character and padding token
@pytest.mark.parametrize("pad_const", (-1, -2))
@pytest.mark.parametrize("y_pred_length", (None, [5, 4]))
def test_ctc_accuracy_function(pad_const, y_pred_length):
    classes_count = 10

    y_true = [[0, 1, 2],
              [4, 2, pad_const]]
    y_pred = tf.one_hot([[classes_count, 0, 1, classes_count, 2] + [classes_count] * 10,
                         [4, classes_count, 5, 5, classes_count] + [classes_count] * 10], classes_count + 1)

    accuracy = ctc_accuracy(y_true, y_pred, y_true_padding_const=pad_const, y_pred_length=y_pred_length)

    assert_array_equal(accuracy, [1, 0])
