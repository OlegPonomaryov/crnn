"""Tests for the _data module"""
import pytest
from numpy.testing import assert_array_equal

from crnn.tools import get_sequences_lengths


@pytest.mark.parametrize("sequences", ([[1, 2, 3, 4],
                                       [4, 2, -1, -1]],))
@pytest.mark.parametrize("padding_constant, expected_lengths", [(None, [4, 4]), (-1, [4, 2])])
def test_get_sequences_lengths(sequences, padding_constant, expected_lengths):
    lengths = get_sequences_lengths(sequences, padding_constant=padding_constant)

    assert_array_equal(lengths, expected_lengths)
