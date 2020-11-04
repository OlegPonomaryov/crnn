"""Tests for the CTCLoss layer"""
import tensorflow as tf

import pytest

from crnn.metrics import CTCAccuracy


# pad_const argument is used to check that CTCAccuracy works if y_true padding const both equal and not equal to the
# ctc_decode()'s unknown character and padding token
@pytest.mark.parametrize("pad_const", (-1, -2))
def test_result(pad_const):
    classes_count = 10

    test_batches = [
        ([[0, 1, 2],
          [4, 2, pad_const]],

         [[0, 1, classes_count, 2, classes_count],
          [classes_count, 4, classes_count, 2, 2]]),

        ([[0, 1, 2],
          [4, 2, pad_const]],

          [[0, 1, classes_count, 2, classes_count],
           [classes_count, 9, classes_count, 2, 2]]),

        ([[0, 1, 2],
          [4, 2, pad_const]],

         [[0, 1, classes_count, 5, classes_count],
          [classes_count, 9, classes_count, 2, 2]])
    ]

    ctc_accuracy = CTCAccuracy(pad_const)

    for y_true, y_pred in test_batches:
        ctc_accuracy.update_state(y_true, tf.one_hot(y_pred, classes_count + 1))
    accuracy = ctc_accuracy.result()

    assert accuracy == 0.5
