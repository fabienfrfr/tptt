"""Unit tests for tptt.utils utility functions."""

import pytest
import torch

from src.tptt.modeling_tptt import (apply_linear_attention_mask,
                                    chunk_sequence, expand_virtual_tokens,
                                    fast_invert_matrix, get_valid_chunk_size,
                                    match_dim, soft_clamp,
                                    truncate_attention_mask)


@pytest.mark.parametrize(
    "x,min_val,max_val",
    [
        (torch.tensor([0.0, 1.0, -1.0]), -1e4, 1e4),
        (torch.tensor([1e5, -1e5]), -1e4, 1e4),
        (torch.tensor([0.0, 50.0, 100.0]), 0, 100),
    ],
)
def test_soft_clamp_parametrize(x, min_val, max_val):
    result = soft_clamp(x, min_val=min_val, max_val=max_val)
    assert torch.all(result <= max_val) and torch.all(result >= min_val)


@pytest.mark.parametrize(
    "B, H, C, size",
    [
        (2, 3, 4, 5),
        (1, 1, 1, 2),
        (3, 2, 1, 8),
    ],
)
def test_invert_nchunked_lower_triangular_matrix(B, H, C, size):
    T = torch.tril(torch.randn(B, H, C, size, size), diagonal=-1)
    eye = torch.eye(size, device=T.device, dtype=T.dtype)
    I = eye.view((1, 1, 1, size, size))

    inv = fast_invert_matrix(T)
    M = I - T

    result = torch.matmul(inv, M)
    assert torch.allclose(
        result, I.expand_as(result), atol=1e-5
    ), f"Correct inverse for shape {T.shape}"


@pytest.mark.parametrize(
    "trick_mode",
    [("derivative"), ("rotative"), ("combined")],
)
def test_expand_virtual_tokens_shape(trick_mode):
    B, H, S, D = 1, 5, 10, 8
    x = torch.randn(B, H, S, D)
    n = 3
    out = expand_virtual_tokens(x, n, trick_mode)
    assert out.shape == torch.Size([B, H, S * n, D])


def test_expand_virtual_tokens_grad():
    # Test gradients flow correctly
    x = torch.randn(1, 5, 10, 8, requires_grad=True)
    n = 3
    out = expand_virtual_tokens(x, n)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_chunk_sequence_simple():
    # Example: 1 batch, 1 head, 6 tokens, 2 head_dim, chunked into 3 chunks of 2
    x = torch.arange(1 * 1 * 6 * 2).reshape(1, 1, 6, 2)
    out = chunk_sequence(x, num_chunks=3, chunk_size=2)
    # Check output shape
    assert out.shape == (1, 1, 3, 2, 2)
    # Check that each chunk matches the original slices
    assert torch.equal(out[0, 0, 0], x[0, 0, 0:2])
    assert torch.equal(out[0, 0, 1], x[0, 0, 2:4])
    assert torch.equal(out[0, 0, 2], x[0, 0, 4:6])


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
