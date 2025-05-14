# pylint: disable=redefined-outer-name
"""Tests for the AttentionOperator module."""

import pytest
import torch

from liza.attention_operator import AttentionOperator


@pytest.fixture
def tensor_dim():
    """Fixture for tensor dimension."""
    return 64


@pytest.fixture
def chunk_size():
    """Fixture for chunk size."""
    return 32


@pytest.fixture
def seq_len():
    """Fixture for sequence length."""
    return 128


@pytest.fixture
def random_tensors(seq_len, tensor_dim):
    """Fixture for random Q, K, V, beta tensors."""
    q = torch.randn(seq_len, tensor_dim)
    k = torch.randn(seq_len, tensor_dim)
    v = torch.randn(seq_len, tensor_dim)
    beta = torch.randn(seq_len, tensor_dim)
    return q, k, v, beta


@pytest.fixture
def operator(tensor_dim):
    """Fixture for AttentionOperator instance."""
    return AttentionOperator(mode="delta_rule", head_dim=tensor_dim)


def test_forward_shape(operator, random_tensors, chunk_size, seq_len, tensor_dim):
    """Test output shape of the forward pass."""
    q, k, v, beta = random_tensors
    o, _ = operator(q, k, v, beta, chunk_size)
    assert o.shape == (seq_len, tensor_dim)


def test_invalid_mode_raises():
    """Test that invalid mode raises ValueError."""
    with pytest.raises(ValueError):
        AttentionOperator(mode="unknown_mode")


def test_chunk_size_1(operator, seq_len, tensor_dim):
    """Test operator with chunk_size=1."""
    q = k = v = beta = torch.ones(seq_len, tensor_dim)
    o, _ = operator(q, k, v, beta, chunk_size=1)
    assert o.shape == (seq_len, tensor_dim)
