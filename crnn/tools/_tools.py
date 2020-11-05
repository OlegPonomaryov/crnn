"""Some useful functions"""
import tensorflow as tf


def get_sequences_lengths(sequences, padding_constant=None):
    """A function to calculate a length of each sequence from the input matrix (excluding padding if it exists)

    :param sequences: A sequences matrix of shape [sequences_count, max_sequence_length]
    :param padding_constant: A padding constant value that shouldn't be included into the lengths of the sequences (None
        if there is no padding), only padding at the end of a sequence is supported
    :return: Length of each sequence without padding
    """
    sequences_shape = tf.shape(sequences)
    sequences_count = sequences_shape[0]
    max_sequence_length = sequences_shape[1]

    if padding_constant is None:
        return tf.fill((sequences_count,), max_sequence_length)
    else:
        not_padding = tf.not_equal(sequences, padding_constant)
        not_padding = tf.cast(not_padding, tf.int32)
        return tf.reduce_sum(not_padding, axis=1)


def get_output_length(padded_input_length, input_lengths, padded_output_length):
    """Calculates size of the output without padding (used to calculate how much columns of a feature map were
        correspond to an actual image and not its padding)

    :param padded_input_length: Input length with padding
    :param input_lengths: Length of each input sample without padding
    :param padded_output_length: Output size with padding
    :return: Length of each output sample without padding
    """
    factor = padded_output_length / padded_input_length
    output_lengths = tf.math.ceil(tf.cast(input_lengths, tf.float32) * tf.cast(factor, tf.float32))
    return tf.cast(output_lengths, tf.int32)
