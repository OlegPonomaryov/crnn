"""Feature sequence extraction layer"""
import tensorflow as tf


class MapToSequence(tf.keras.layers.Layer):
    """Feature sequence extraction for CRNN as described in (Shi et al., 2015). Specifically, the feature vector of the
    i-th timestep of the output is the concatenation of the i-th columns of all the maps (channels) of the input."""

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        samples = inputs_shape[0]
        columns = inputs_shape[2]

        output = tf.transpose(inputs, [0, 2, 1, 3])
        output = tf.reshape(output, [samples, columns, -1])

        # If executed in the graph mode, tf.transpose() doesn't set the shape attribute of its output. However, we need
        # at least the last dimension to be set in order to pass the output of the MapToSequence layer to a recurrent
        # layer, so we set the shape manually.
        if output.shape[-1] is None:
            # It is recommended to use tf.shape() in the graph mode, but it returns tensors that aren't accepted by the
            # set_shape(). On the other hand, we mostly need shape[1], which is height and is normalized for all
            # images, and shape[3], which is the number of feature maps and is defined by the last convolutional layer's
            # hyperparameter, so both this values are known before the call() function tracing and should be present in
            # the shape attribute. Other dimensions can be None, this won't cause any errors.
            output.set_shape([inputs.shape[0], inputs.shape[2], inputs.shape[1] * inputs.shape[3]])

        return output
