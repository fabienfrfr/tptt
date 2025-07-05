# pylint: disable=redefined-outer-name
"""Tests for the AttentionOperator module."""

import pytest

from src.tptt.modeling_tptt import LinearAttention


def test_forward_shape(
    operator,
    random_qkv_tensors,
    chunk_size,
    seq_len,
    batch_size,
    num_heads,
    head_dim,
):
    """Test output shape of the forward pass."""
    q, k, v, beta = random_qkv_tensors
    o = operator(q, k, v, beta=beta, chunk_size=chunk_size)
    assert o.shape == (batch_size, num_heads, seq_len, head_dim)


def test_attention_operator_raises_on_unknown_mode(random_qkv_tensors):
    with pytest.raises(ValueError):
        op = LinearAttention(layer_idx=1, mode="not_a_mode")
        q, k, v, beta = random_qkv_tensors
        op(q, k, v, beta)


def test_chunk_delta_rule_forward_computation(
    random_qkv_tensors, batch_size, num_heads, seq_len, head_dim
):
    q, k, v, beta = random_qkv_tensors
    chunk_size = 8
    beta = beta[0]  # Only key or value gate is used in delta rule forward
    out, _ = LinearAttention.chunk_delta_product_forward(q, k, v, beta, chunk_size)
    assert out.shape == (batch_size, num_heads, seq_len, head_dim)


def test_chunk_delta_product_forward_computation(
    random_qkv_tensors, batch_size, num_heads, seq_len, head_dim
):
    q, k, v, beta = random_qkv_tensors
    n = 2
    chunk_size = 8
    beta = beta[0]  # Only key or value gate is used in delta rule forward
    out, _ = LinearAttention.chunk_delta_product_forward(q, k, v, beta, chunk_size, n)
    assert out.shape == (batch_size, num_heads, seq_len, head_dim)
