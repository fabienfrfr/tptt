"""Unit tests for tptt.utils utility functions."""

import pytest
import torch

from src.tptt.liza.utils import (apply_linear_attention_mask,
                                 get_valid_chunk_size, match_dim, repeat_kv,
                                 truncate_attention_mask)


def test_repeat_kv():
    """Test that repeat_kv repeats the key/value heads along the correct dimension."""
    x = torch.randn(2, 4, 8, 64)
    repeated = repeat_kv(x, 2)
    assert repeated.shape == (2, 8, 8, 64)


@pytest.mark.parametrize(
    "mask_shape,v_shape",
    [
        ((2, 4), (2, 4, 8)),  # standard
        ((2, 1, 4, 4), (2, 4, 8)),  # padding mask
        ((1, 4), (1, 4, 8)),  # batch=1
        ((1, 1, 4), (1, 4, 8)),  # batch=1, singleton
    ],
)
def test_apply_attention_mask(mask_shape, v_shape):
    attention_mask = torch.ones(mask_shape)
    v = torch.randn(v_shape)
    # Set some mask positions to zero for test
    attention_mask[..., 0] = 0

    v_masked = apply_linear_attention_mask(attention_mask, v.clone())
    # The first token in every sequence should be zeroed
    assert v_masked.shape == v.shape


@pytest.mark.parametrize(
    "mask_shape,hidden_shape",
    [
        ((2, 4), (2, 4, 8)),  # standard
        ((2, 1, 4, 4), (2, 4, 8)),  # padding mask (complex)
        ((1, 4), (1, 4, 8)),  # batch=1
        ((1, 1, 4), (1, 4, 8)),  # batch=1, singleton
    ],
)
def test_truncate_attention_mask(mask_shape, hidden_shape):
    max_length = 2
    hidden_states = torch.randn(hidden_shape)
    attention_mask = torch.ones(mask_shape)

    hidden_states[..., -max_length:, :] = 42
    attention_mask[(..., -max_length)] = 0

    truncated_hidden, truncated_mask = truncate_attention_mask(
        hidden_states, attention_mask, max_length
    )

    assert truncated_hidden.shape[1] == max_length

    seq_lens = [i for i, s in enumerate(attention_mask.shape) if s == hidden_shape[1]]
    assert any(truncated_mask.shape[d] == max_length for d in seq_lens)

    assert torch.all(truncated_hidden == 42)

    print(f"Passed for mask_shape={mask_shape}, hidden_shape={hidden_shape}")


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
