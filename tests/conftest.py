# pylint: disable=redefined-outer-name
"""Tests for the AttentionOperator module."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from src.tptt.configuration_tptt import TpttConfig
from src.tptt.modeling_tptt import (
    LCache,
    LinearAttention,
    LinearAttentionOp,
    LiZAttention,
    TpttModel,
)


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
def num_q_heads():
    """Fixture for num_q_heads."""
    return 8


@pytest.fixture
def num_kv_heads():
    """Fixture for num_kv_heads."""
    return 2


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
    beta = (
        torch.randn(batch_size, num_heads, seq_len, head_dim),
        torch.randn(batch_size, num_heads, seq_len, head_dim),
    )
    return q, k, v, beta


@pytest.fixture
def random_hidden_tensor(batch_size, seq_len, tensor_dim):
    """Fixture for random hidden_states tensors."""
    return torch.randn(batch_size, seq_len, tensor_dim)


@pytest.fixture
def attention_mask(random_hidden_tensor, seq_len, batch_size):
    """Fixture for attention_mask tensors."""
    return torch.ones(
        batch_size,
        seq_len,
        dtype=random_hidden_tensor.dtype,
        device=random_hidden_tensor.device,
    )


@pytest.fixture
def operator():
    """Fixture for AttentionOperator instance."""
    return LinearAttentionOp(layer_idx=1, operator_mode="delta_rule")


@pytest.fixture
def dummy_base_attn(tensor_dim, num_heads, head_dim):
    """Fixture for a dummy base attention module."""

    class DummyAttention(nn.Module):
        """Minimal dummy attention module with shared projections."""

        def __init__(self):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.q_proj = nn.Linear(tensor_dim, num_heads * head_dim)
            self.k_proj = nn.Linear(tensor_dim, num_heads * head_dim)
            self.v_proj = nn.Linear(tensor_dim, num_heads * head_dim)
            self.o_proj = nn.Linear(num_heads * self.head_dim, tensor_dim)

        def forward(self, x, **kwargs):
            # Simulate output of a standard attention module
            # Output: (batch, seq_len, tensor_dim), attn_weights=None
            return torch.randn(x.shape[0], x.shape[1], x.shape[2]), None

    return DummyAttention()


@pytest.fixture
def dummy_fused_qkv_attn(tensor_dim, num_q_heads, num_kv_heads, head_dim):
    """Fixture for a dummy attention module with fused QKV projection (OpenELM/Falcon style)."""

    class DummyFusedQKVAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_q_heads = num_q_heads
            self.num_k_heads = num_kv_heads
            self.num_v_heads = num_kv_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.qkv_proj = nn.Linear(
                tensor_dim, num_q_heads * head_dim + 2 * num_kv_heads * head_dim
            )
            self.out_proj = nn.Linear(num_q_heads * head_dim, tensor_dim)

        def forward(self, x, **kwargs):
            # Simulate output of a standard attention module
            return torch.randn(x.shape[0], x.shape[1], x.shape[2]), None

    return DummyFusedQKVAttention()


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
def liza_qkv_attention(dummy_fused_qkv_attn, dummy_config):
    """Fixture for AttentionOperator with qkv."""
    return LiZAttention(dummy_fused_qkv_attn, 0, dummy_config)


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
    return LCache()


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


@pytest.fixture
def dummy_tptt_model(dummy_tptt_config, dummy_model, dummy_tokenizer, cache, mocker):
    # Patch les dépendances internes du constructeur TpttModel
    mocker.patch(
        "src.tptt.modeling_tptt.AutoModelForCausalLM.from_pretrained",
        return_value=dummy_model,
    )
    mocker.patch(
        "src.tptt.modeling_tptt.AutoTokenizer.from_pretrained",
        return_value=dummy_tokenizer,
    )
    mocker.patch(
        "src.tptt.injection.inject_linear_attention", return_value=(dummy_model, cache)
    )
    return TpttModel(dummy_tptt_config)


class DummyConfig(PretrainedConfig):
    def __init__(self):
        super().__init__()
        self.vocab_size = 50257


class DummyModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config=None):
        if config is None:
            config = DummyConfig()
        super().__init__(config)
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    def forward(self, *args, **kwargs):
        return torch.ones((1, 1))

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


@pytest.fixture
def dummy_pipeline_components():
    model = DummyModel()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


@pytest.fixture
def linear_attention():
    hidden_dim = 32
    num_heads = 4
    attn = LinearAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        dropout=0.0,
        padding_side="right",
    )
    return attn


@pytest.fixture
def bidirectional_linear_attention():
    hidden_dim = 32
    num_heads = 4
    attn = LinearAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        dropout=0.0,
        padding_side="right",
        bidirectional=True,
    )
    return attn
