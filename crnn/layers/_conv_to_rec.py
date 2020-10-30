"""Feature sequence extraction layer"""
import tensorflow as tf


class ConvToRec(tf.keras.layers.Layer):
    """Feature sequence extraction for CRNN as described in (Shi et al., 2015). Specifically, the feature vector of the
    i-th timestep of the output is the concatenation of the i-th columns of all the maps (channels) of the input."""

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        samples = inputs_shape[0]
        columns = inputs_shape[2]

        x = tf.transpose(inputs, [0, 2, 1, 3])
        x = tf.reshape(x, [samples, columns, -1])

        return x
