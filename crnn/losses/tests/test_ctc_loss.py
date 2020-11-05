"""Tests for the CTC loss"""
import tensorflow as tf

import pytest
from numpy.testing import assert_array_equal, assert_array_less

from crnn.losses import ctc_loss


def test_ctc_loss__padded_and_not_padded_labels():
    classes_count = 10
    pad_const = -1
    y_true = [[4, 2]]
    y_true_padded = [[4, 2, pad_const, pad_const]]
    y_pred = tf.one_hot([[classes_count, 4, classes_count, 2, 2]], classes_count + 1)

    loss_without_padding = ctc_loss(y_true, y_pred)
    loss_with_padding = ctc_loss(y_true_padded, y_pred, y_true_padding_const=pad_const)

    # Padding of the labels shouldn't affect the loss value
    assert_array_equal(loss_without_padding, loss_with_padding)


@pytest.mark.parametrize("y_pred_length", (None, [5, 4]))
def test_ctc_loss__output_values_are_correct(y_pred_length):
    classes_count = 10
    pad_const = -1
    y_true = [[0, 1, 2],
              [4, 2, pad_const]]
    y_pred_good = tf.one_hot([[classes_count, 0, 1, classes_count, 2] + [classes_count] * 10,
                              [4, classes_count, 2, 2, classes_count] + [classes_count] * 10], classes_count + 1)
    y_pred_bad = tf.one_hot([[classes_count, 9, 1, classes_count, 2] + [classes_count] * 10,
                             [4, classes_count, 5, 5, classes_count] + [classes_count] * 10], classes_count + 1)

    good_loss = ctc_loss(y_true, y_pred_good, y_true_padding_const=pad_const, y_pred_length=y_pred_length)
    bad_loss = ctc_loss(y_true, y_pred_bad, y_true_padding_const=pad_const, y_pred_length=y_pred_length)

    # A sanity check that a bad prediction produces a much higher loss than the good one
    assert_array_less(1E6, tf.reduce_mean(bad_loss) / tf.reduce_mean(good_loss))
