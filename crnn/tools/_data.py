"""Tools for working with input and output data"""
import tensorflow as tf


def get_sequences_lengths(sequences, padding_constant=None):
    """A function to calculate a length of each sequence from the input matrix (excluding padding if it exists)

    :param sequences: A sequences matrix of shape [sequences_count, max_sequence_length]
    :param padding_constant: A padding constant value that shouldn't be included into the lengths of the sequences (None
        if there is no padding), only padding at the end of a sequence is supported
    """
    sequences_count, max_sequence_length = tf.shape(sequences)
    if padding_constant is None:
        return tf.fill((sequences_count,), max_sequence_length)
    else:
        not_padding = tf.not_equal(sequences, padding_constant)
        not_padding = tf.cast(not_padding, tf.int32)
        return tf.reduce_sum(not_padding, axis=1)
