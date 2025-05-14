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
    return 16


@pytest.fixture
def tensor_dim():
    """Fixture for tensor dimension."""
    return 64


@pytest.fixture
def chunk_size():
    """Fixture for chunk size."""
    return 32


@pytest.fixture
def batch_size():
    """Fixture for batch_size."""
    return 2


@pytest.fixture
def num_heads():
    """Fixture for num_heads."""
    return 4


@pytest.fixture
def head_dim():
    """Fixture for num_heads."""
    return 8


@pytest.fixture
def random_tensors(batch_size, num_heads, seq_len, head_dim):
    """Fixture for random Q, K, V, beta tensors."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    beta = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return q, k, v, beta


@pytest.fixture
def operator(tensor_dim):
    """Fixture for AttentionOperator instance."""
    return AttentionOperator(mode="delta_rule", head_dim=tensor_dim)
