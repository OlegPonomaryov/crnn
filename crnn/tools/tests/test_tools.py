"""Tests for the _tools module"""
import numpy as np

import pytest
from numpy.testing import assert_array_equal

from crnn.tools import get_sequences_lengths, get_output_length


@pytest.mark.parametrize("sequences", ([[1, 2, 3, 4],
                                       [4, 2, -1, -1]],))
@pytest.mark.parametrize("padding_constant, expected_lengths", [(None, [4, 4]), (-1, [4, 2])])
def test_get_sequences_lengths(sequences, padding_constant, expected_lengths):
    lengths = get_sequences_lengths(sequences, padding_constant=padding_constant)

    assert_array_equal(lengths, expected_lengths)


def test_get_output_length():
    input_size = 100
    input_lengths = np.asarray([80, 60, 50])
    output_size = 25

    output_lengths = get_output_length(input_size, input_lengths, output_size)

    assert_array_equal(output_lengths, [20, 15, 13])
