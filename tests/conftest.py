# pylint: disable=redefined-outer-name
"""Tests for the AttentionOperator module."""

import pytest
import torch
from torch import nn

from src.tptt.liza.mapping import AttentionOperator


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


@pytest.fixture
def dummy_base_attn():
    """Fixture for a dummy base attention module."""

    class DummyAttention(nn.Module):
        """Minimal dummy attention module with shared projections."""

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)

        def forward(self, x, **kwargs):  # pylint: disable=unused-argument
            """Simulate output of a standard attention module."""
            return torch.randn(x.shape[0], x.shape[1], 64), None

    return DummyAttention()


@pytest.fixture
def dummy_config():
    """Fixture for a dummy configuration object."""

    class DummyConfig:  # pylint: disable=too-few-public-methods
        """Minimal dummy config for LiZAttention."""

        hidden_size = 64
        num_attention_heads = 8
        num_key_value_heads = 4
        attention_dropout = 0.1
        head_dim = 8

    return DummyConfig


@pytest.fixture
def dummy_decoder(dummy_base_attn):
    """Fixture for a dummy decoder module."""

    class DummyDecoder(nn.Module):
        """Dummy model containing a self-attention module."""

        def __init__(self):
            super().__init__()
            self.self_attn = dummy_base_attn

    return DummyDecoder
