"""CTC decode layer"""
import tensorflow as tf


class CTCDecode(tf.keras.layers.Layer):
    """Decodes a CTC sequence into a standard sequence of labels or characters

    :param label_to_char: A StaticHashTable to convert labels to characters
    :param name: String name of the layer
    """

    def __init__(self, label_to_char=None, name=None):
        super().__init__(name=name)
        self.label_to_char = label_to_char

    def call(self, inputs):
        labels = tf.keras.backend.ctc_decode(inputs, tf.fill((inputs.shape[0]), inputs.shape[1]))[0][0]
        labels = tf.cast(labels, tf.int32)
        return self.label_to_char.lookup(labels)
