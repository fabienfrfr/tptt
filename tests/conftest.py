# pylint: disable=redefined-outer-name, too-many-arguments, too-many-positional-arguments

"""Conftest for the tppt module."""

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
def random_qkv_tensors(attention_dims):
    """Generate random q, k, v tensors and two beta tensors (tuple)."""
    batch_size, num_head, seq_len, head_dim = attention_dims.values()
    q = torch.randn(batch_size, num_head, seq_len, head_dim)
    k = torch.randn(batch_size, num_head, seq_len, head_dim)
    v = torch.randn(batch_size, num_head, seq_len, head_dim)
    alpha = torch.ones(batch_size, num_head, seq_len, 1)  # head_dim ?
    beta = torch.sigmoid(torch.randn(batch_size, num_head, seq_len, 1))  # head_dim ?
    return (q, k, v, alpha, beta)


@pytest.fixture
def random_qkv_expended_tensors(attention_dims):
    """Generate expanded random q, k, v tensors and two beta tensors (tuple)."""
    batch_size, num_head, seq_len, head_dim = attention_dims.values()
    n_orders = 2
    q = torch.randn(batch_size, num_head, seq_len, n_orders, head_dim)
    k = torch.randn(batch_size, num_head, seq_len, n_orders, head_dim)
    v = torch.randn(batch_size, num_head, seq_len, n_orders, head_dim)
    # no head_dim for alpha/beta in expanded mode (not mandatory ?)
    alpha = torch.ones(batch_size, num_head, seq_len, n_orders, 1)
    beta = torch.sigmoid(torch.randn(batch_size, num_head, seq_len, n_orders, 1))
    return (q, k, v, alpha, beta)


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

        def forward(self, x, **kwargs):  # pylint: disable=unused-argument
            """Simulate output of a standard attention module"""
            # Output: (batch, seq_len, tensor_dim), attn_weights=None
            return torch.randn(x.shape[0], x.shape[1], x.shape[2]), None

    return DummyAttention()


@pytest.fixture
def dummy_fused_qkv_attn(tensor_dim, num_q_heads, num_kv_heads, head_dim):
    """Fixture for a dummy attention module with fused QKV projection (OpenELM/Falcon style)."""

    class DummyFusedQKVAttention(nn.Module):
        """Minimal dummy qkv attention"""

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

        def forward(self, x, **kwargs):  # pylint: disable=unused-argument
            """Simulate output of a standard attention module"""
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

    class DummyConfig:  # pylint: disable=too-few-public-methods
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

    class DummyDecoder(nn.Module):  # pylint: disable=abstract-method
        """Dummy model containing a self-attention module."""

        def __init__(self):
            super().__init__()
            self.self_attn = dummy_base_attn

    return DummyDecoder


@pytest.fixture
def cache():
    """Basic cache pre-imported"""
    return LCache()


@pytest.fixture
def dummy_tptt_config():
    """Basic config"""
    return TpttConfig(model_name="test-model")


@pytest.fixture
def dummy_tokenizer():
    """Basic tokenizer"""
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
    """Dummy Self Attn model"""
    model = MagicMock()
    model.named_modules.return_value = [("layer.self_attn", MagicMock())]
    model.generate.return_value = [[0, 1, 2]]
    model.save_pretrained = MagicMock()
    return model


@pytest.fixture
def dummy_tptt_model(dummy_tptt_config, dummy_model, dummy_tokenizer, cache, mocker):
    """Patch les dépendances internes du constructeur TpttModel"""
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
    """Basic mock tptt config"""

    def __init__(self):
        super().__init__()
        self.vocab_size = 50257


class DummyModel(PreTrainedModel):
    """Basick import mock model"""

    config_class = DummyConfig

    def __init__(self, config=None):
        if config is None:
            config = DummyConfig()
        super().__init__(config)
        self._device = torch.device("cpu")

    @property
    def device(self):
        """get device"""
        return self._device

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        """minimal forward"""
        return torch.ones((1, 1))

    def generate(self, **kwargs):  # pylint: disable=unused-argument
        """minimal generation"""
        return torch.tensor([[1, 2, 3, 4]])


@pytest.fixture
def dummy_pipeline_components():
    """minimal pipeline"""
    model = DummyModel()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


@pytest.fixture
def deltarule_attention():
    """minimal deltarule attention"""
    hidden_dim = 32
    num_heads = 4
    recurrent_config = {
        "order": 1,
        "alpha_gate": "c",
        "beta_gate": "k",
        "linear": True,
        "trick": "cte",
    }
    attn = LinearAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        dropout=0.0,
        padding_side="right",
        recurrent_config=recurrent_config,
    )
    return attn


@pytest.fixture
def linear_attention():
    """minimal linear attention"""
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
    """minimal bidirectional attention"""
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


@pytest.fixture
def make_fake_state(keys=("a.lora_A.weight", "b.lora_B.weight")):
    """Fake state dict for safetensors."""
    return {k: f"tensor_{k}" for k in keys}


@pytest.fixture
def make_fake_model(state_keys):
    """Return a MagicMock-like model with required API."""
    model = MagicMock()
    # Returns your input keys (simulate expected keys in model)
    model.state_dict.return_value = {k: None for k in state_keys}

    # load_state_dict returns lists of missing/unexpected keys
    def load_state_dict(sd, strict, assign):
        # Return missing all "lora" that not in state_dict, and unexpected all that don't match
        missing = [k for k in model.state_dict() if k not in sd]
        unexpected = [k for k in sd if k not in model.state_dict()]
        return (missing, unexpected)

    model.load_state_dict.side_effect = load_state_dict
    return model


@pytest.fixture(autouse=True)
def patch_ensure_stability(monkeypatch):
    """Patch ensure_stability globally to be identity for simplicity."""
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.ensure_stability",
        lambda x, **kwargs: x,
    )


@pytest.fixture
def attention_dims():
    """Return dimension dict: batch, num_head, seq_len, head_dim."""
    return dict(batch=2, num_heads=3, seq_len=12, head_dim=8)


@pytest.fixture
def dummy_cache():
    """Simple dict-like cache compatible with LinearAttentionOp."""

    class DummyCache(dict):
        def update(self, key, recurrent_state=None, qkvg=None):
            entry = {"recurrent_state": recurrent_state}
            if qkvg is not None:
                entry["qkvg"] = qkvg
            self[key] = entry

        def __getitem__(self, key):
            return dict.get(self, key, None)

    return DummyCache()


@pytest.fixture
def linear_operator():
    """A default LinearAttentionOp with quadratic operator and working cache."""
    return LinearAttentionOp(
        max_chunk_size=6,
        linear_precision=torch.float32,
    )
