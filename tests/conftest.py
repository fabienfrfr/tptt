# pylint: disable=redefined-outer-name
"""Tests for the AttentionOperator module."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from src.tptt.liza.mapping_func import AttentionOperator
from src.tptt.liza.memory_gate import LiZAttention
from src.tptt.modeling_tptt import TpttConfig
from src.tptt.utils import Cache


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
    """Fixture for dim heads."""
    return 8


@pytest.fixture
def num_key_value_heads():
    """Fixture for num_key_value_heads."""
    return 4


@pytest.fixture
def attention_dropout():
    """Fixture for attention_dropout."""
    return 0.1


@pytest.fixture
def max_length():
    """Fixture for max_length."""
    return 2048


@pytest.fixture
def tensor_dim(head_dim, num_heads):
    """Fixture for tensor dimension."""
    return head_dim * num_heads


@pytest.fixture
def random_qkv_tensors(batch_size, num_heads, seq_len, head_dim):
    """Fixture for random Q, K, V, beta tensors."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    beta = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return q, k, v, beta


@pytest.fixture
def random_hidden_tensor(batch_size, seq_len, tensor_dim):
    """Fixture for random hidden_states tensors."""
    return torch.randn(batch_size, seq_len, tensor_dim)


@pytest.fixture
def operator(tensor_dim):
    """Fixture for AttentionOperator instance."""
    return AttentionOperator(mode="delta_rule", head_dim=tensor_dim)


@pytest.fixture
def dummy_base_attn(tensor_dim):
    """Fixture for a dummy base attention module."""

    class DummyAttention(nn.Module):
        """Minimal dummy attention module with shared projections."""

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(tensor_dim, tensor_dim)
            self.k_proj = nn.Linear(tensor_dim, tensor_dim)
            self.v_proj = nn.Linear(tensor_dim, tensor_dim)
            self.o_proj = nn.Linear(tensor_dim, tensor_dim)

        def forward(self, x, **kwargs):  # pylint: disable=unused-argument
            """Simulate output of a standard attention module."""
            return torch.randn(x.shape[0], x.shape[1], tensor_dim), None

    return DummyAttention()


@pytest.fixture
def dummy_config(
    tensor_dim,
    num_heads,
    num_key_value_heads,
    attention_dropout,
    head_dim,
    max_length,
):
    """Fixture for a dummy configuration object."""

    class DummyConfig:
        """Minimal dummy config for LiZAttention."""

        def __init__(
            self,
            tensor_dim,
            num_heads,
            num_key_value_heads,
            attention_dropout,
            head_dim,
            max_length,
        ):
            self.hidden_size = tensor_dim
            self.num_attention_heads = num_heads
            self.num_key_value_heads = num_key_value_heads
            self.attention_dropout = attention_dropout
            self.head_dim = head_dim
            self.max_length = max_length

    return DummyConfig(
        tensor_dim,
        num_heads,
        num_key_value_heads,
        attention_dropout,
        head_dim,
        max_length,
    )


@pytest.fixture
def liza_attention(dummy_base_attn, dummy_config):
    """Fixture for AttentionOperator instance."""
    return LiZAttention(dummy_base_attn, 1, dummy_config)


@pytest.fixture
def dummy_decoder(dummy_base_attn):
    """Fixture for a dummy decoder module."""

    class DummyDecoder(nn.Module):
        """Dummy model containing a self-attention module."""

        def __init__(self):
            super().__init__()
            self.self_attn = dummy_base_attn

    return DummyDecoder


@pytest.fixture
def cache():
    return Cache()


@pytest.fixture
def cache_with_max_length(max_length):
    return Cache(max_length=max_length)


@pytest.fixture
def dummy_tptt_config():
    return TpttConfig(model_name="test-model")


@pytest.fixture
def dummy_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_tensors = "pt"
    tokenizer.decode.return_value = "Generated text"
    tokenizer.return_value = {
        "input_ids": [0, 1, 2],
        "attention_mask": [1, 1, 1],
    }
    return tokenizer


@pytest.fixture
def dummy_model():
    model = MagicMock()
    model.named_modules.return_value = [("layer.self_attn", MagicMock())]
    model.generate.return_value = [[0, 1, 2]]
    model.save_pretrained = MagicMock()
    return model
