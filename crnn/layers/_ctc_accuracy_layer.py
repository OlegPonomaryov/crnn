"""CTC accuracy layer"""
import tensorflow as tf

from crnn.tools import get_output_length
from crnn.metrics import ctc_accuracy


class CTCAccuracyLayer(tf.keras.layers.Layer):
    """Adds CTC accuracy metric to a model and returns unmodified y_true

    :param y_true_padding_const: A value that was used for the true labels sequences padding (None if there is no
        padding, only padding at the end of a sequence is supported)
    :param name: String name of the layer
    """

    def __init__(self, y_true_padding_const=None, name=None):
        super().__init__(name=name)
        self.y_true_padding_const = y_true_padding_const

    def call(self, y_true, y_pred, input=None, input_lengths=None):
        if input is not None and input_lengths is not None:
            y_pred_length = get_output_length(tf.shape(input)[2], input_lengths, tf.shape(y_pred)[1])
        else:
            y_pred_length = None

        accuracy = ctc_accuracy(
            y_true, y_pred, y_true_padding_const=self.y_true_padding_const, y_pred_length=y_pred_length)
        accuracy = tf.cast(accuracy, tf.float32)
        self.add_metric(accuracy, name="accuracy")
        return y_pred
