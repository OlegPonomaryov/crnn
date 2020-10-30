"""CTC loss"""
import tensorflow as tf

from crnn.tools import get_sequences_lengths


class CTCLoss(tf.keras.losses.Loss):
    """A wrapper around the tf.keras.backend.ctc_batch_cost() function to allow using it as a tf.keras.losses.Loss()
    object

    :param padding_constant: A padding constant value that shouldn't be included into the lengths of the true labels
        sequences (None if there is no padding), only padding at the end of a sequence is supported
    """

    def __init__(self, padding_constant=None):
        super().__init__()
        self.padding_constant = padding_constant

    def call(self, y_true, y_pred):
        y_true_length = get_sequences_lengths(y_true, self.padding_constant)
        y_true_length = tf.expand_dims(y_true_length, axis=1)

        y_pred_shape = tf.shape(y_pred)
        y_pred_length = tf.fill((y_pred_shape[0], 1), y_pred_shape[1])

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, y_pred_length, y_true_length)
        return tf.squeeze(loss)
