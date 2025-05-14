"""Unit tests for liza.utils utility functions."""

import torch

from liza.utils import get_valid_chunk_size, match_dim, repeat_kv


def test_repeat_kv():
    """Test that repeat_kv repeats the key/value heads along the correct dimension."""
    x = torch.randn(2, 4, 8, 64)
    repeated = repeat_kv(x, 2)
    assert repeated.shape == (2, 8, 8, 64)


def test_match_dim_expand():
    """Test that match_dim expands the specified dimension correctly."""
    x = torch.randn(1, 10, 32)
    matched = match_dim(x, 1, 20)
    assert matched.shape == (1, 20, 32)


def test_match_dim_reduce():
    """Test that match_dim reduces the specified dimension correctly."""
    x = torch.randn(1, 10, 32)
    matched = match_dim(x, 1, 5)
    assert matched.shape == (1, 5, 32)


def test_get_valid_chunk_size():
    """Test get_valid_chunk_size returns the expected chunk size."""
    assert get_valid_chunk_size(127, 64) == 1
    assert get_valid_chunk_size(128, 64) == 64
    assert get_valid_chunk_size(120, 64) == 60


from liza.utils import match_dim


def test_match_dim_interpolate():
    x = torch.randn(2, 4, 8)
    # Expand dim=1 from 4 to 7
    y = match_dim(x, 1, 7)
    assert y.shape == (2, 7, 8)


def test_match_dim_projection():
    x = torch.randn(2, 7, 8)
    # Reduce dim=1 from 7 to 4
    y = match_dim(x, 1, 4)
    assert y.shape == (2, 4, 8)
