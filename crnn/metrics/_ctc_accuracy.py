"""CTC accuracy"""
import tensorflow as tf


# An unknown character token returned by ctc_decode(), which also serves as a padding constant
DECODED_PADDING_CONSTANT = -1


class CTCAccuracy(tf.keras.metrics.Mean):
    """Calculates how often a predicted text fully matches the corresponding true text

    :param y_true_padding_const: A value that was used for the true labels sequences padding (None if there is no
        padding, only padding at the end of a sequence is supported)
    :param name: A string name of the metric instance.
    :param dtype: A data type of the metric result.
    """

    def __init__(self, y_true_padding_const=None, name="accuracy", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.y_true_padding_const = y_true_padding_const

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(ctc_accuracy(
            y_true, y_pred, y_true_padding_const=self.y_true_padding_const), sample_weight)


def ctc_accuracy(y_true, y_pred, y_true_padding_const=None, y_pred_length=None):
    """Calculates how often a predicted text fully matches the corresponding true text

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_true_padding_const: A value that was used for the true labels sequences padding (None if there is no
        padding, only padding at the end of a sequence is supported)
    :param y_pred_length: Length of predictions without padding
    :return: A 1D tensor of 0 and 1 indicating whether predictions for each samples matched true values
    """
    y_pred_shape = tf.shape(y_pred)
    y_true_shape = tf.shape(y_true)

    if y_pred_length is None:
        y_pred_length = tf.fill((y_pred_shape[0],), y_pred_shape[1])
    else:
        max_y_pred_length = tf.reduce_max(y_pred_length)
        if max_y_pred_length < y_pred_shape[1]:
            y_pred = y_pred[:, :max_y_pred_length, :]
            y_pred_shape = tf.shape(y_pred)

    y_pred = tf.keras.backend.ctc_decode(y_pred, y_pred_length)[0][0]
    y_pred_shape = y_pred_shape[:-1]  # y_pred shape after the decoding loses the last dimension (classes count)

    # If y_true is padded, update it to use the same padding constant as y_pred_decoded
    if y_true_padding_const is not None and y_true_padding_const != DECODED_PADDING_CONSTANT:
        y_true = tf.where(tf.not_equal(y_true, y_true_padding_const),
                          y_true, tf.fill(y_true_shape, DECODED_PADDING_CONSTANT))

    # Pad y_true or y_pred so that they have the same max sequence length
    if y_true_shape[1] < y_pred_shape[1]:
        y_true = tf.pad(y_true, paddings=[[0, 0], [0, y_pred_shape[1] - y_true_shape[1]]],
                        constant_values=DECODED_PADDING_CONSTANT)
        y_true_shape = y_pred_shape
    elif y_pred_shape[1] < y_true_shape[1]:
        y_pred = tf.pad(y_pred, paddings=[[0, 0], [0, y_true_shape[1] - y_pred_shape[1]]],
                        constant_values=DECODED_PADDING_CONSTANT)
        y_pred_shape = y_true_shape

    # ctc_decode() returns y_pred of int64 type and if y_true is int32, that will cause an error in tf.equal(), so
    # we need to cast y_pred to the same type
    if y_pred.dtype != y_true.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    matching_labels_count = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int32), axis=1)
    # A prediction considered correct if all its labels match the corresponding true sequence
    correct_predictions = tf.equal(matching_labels_count, y_true_shape[1])

    return tf.cast(correct_predictions, tf.int32)
