"""CTC loss"""
import tensorflow as tf

from crnn.tools import get_sequences_lengths


class CTCLoss(tf.keras.losses.Loss):
    """Calculates CTC loss

    :param y_true_padding_const: A value that was used for the true labels sequences padding (None if there is no
        padding, only padding at the end of a sequence is supported)
    :param reduction: Reduction argument for the base tf.keras.losses.Loss class
    :param name: Optional name for the op
    """

    def __init__(self, y_true_padding_const=None, reduction=tf.losses.Reduction.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.y_true_padding_const = y_true_padding_const

    def call(self, y_true, y_pred):
        return ctc_loss(y_true, y_pred, y_true_padding_const=self.y_true_padding_const)


def ctc_loss(y_true, y_pred, y_true_padding_const=None, y_pred_length=None):
    """Calculates CTC loss

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_true_padding_const: A value that was used for the true labels sequences padding (None if there is no
        padding, only padding at the end of a sequence is supported)
    :param y_pred_length: Length of predictions without padding
    :return: A losss for each sample
    """
    y_true_length = get_sequences_lengths(y_true, y_true_padding_const)
    y_true_length = tf.expand_dims(y_true_length, axis=1)

    if y_pred_length is None:
        y_pred_shape = tf.shape(y_pred)
        y_pred_length = tf.fill((y_pred_shape[0], 1), y_pred_shape[1])
    else:
        y_pred_length = tf.expand_dims(y_pred_length, axis=1)

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, y_pred_length, y_true_length)
    return tf.squeeze(loss)
