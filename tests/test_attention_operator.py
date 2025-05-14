# pylint: disable=redefined-outer-name
"""Tests for the AttentionOperator module."""

import pytest
import torch

from liza.attention_operator import AttentionOperator


def test_forward_shape(
    operator,
    random_tensors,
    chunk_size,
    seq_len,
    batch_size,
    num_heads,
    head_dim,
):
    """Test output shape of the forward pass."""
    q, k, v, beta = random_tensors
    o, _ = operator(q, k, v, beta=beta, chunk_size=chunk_size)
    assert o.shape == (batch_size * num_heads * seq_len, head_dim)


def test_attention_operator_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        op = AttentionOperator(mode="not_a_mode")
        op(None, None, None)


def test_chunk_delta_rule_forward_computation(batch_size, num_heads, seq_len, head_dim):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    beta = torch.randn(batch_size, num_heads, seq_len, head_dim)
    chunk_size = 8
    out, _ = AttentionOperator.chunk_delta_rule_forward(q, k, v, beta, chunk_size)
    assert out.shape == (batch_size * num_heads * seq_len, head_dim)
