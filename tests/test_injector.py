"""Unit test for inject_linear_attention replacing a module with LiZAttention."""

import torch
from torch import nn

from liza.injector import inject_linear_attention
from liza.linear_attention import LiZAttention


class DummySelfAttn(nn.Module):
    """Dummy self-attention module with projection layers."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)

    def forward(self, x, **kwargs):  # pylint: disable=unused-argument
        """Simulate the output of a standard attention module."""
        return torch.randn(x.shape[0], x.shape[1], 64), None


class DummyModel(nn.Module):
    """Dummy model containing a self-attention module."""

    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttn()


class DummyConfig:  # pylint: disable=too-few-public-methods
    """Minimal dummy config for LiZAttention."""

    hidden_size = 64
    num_attention_heads = 8
    num_key_value_heads = 4
    attention_dropout = 0.1
    head_dim = 8


def test_inject_linear_attention_replaces_module():
    """Test that inject_linear_attention replaces the target module with LiZAttention."""
    model = DummyModel()
    injected = inject_linear_attention(
        model, DummyConfig(), target_modules=["self_attn"], mag_weight=0.7
    )
    assert isinstance(injected.self_attn, LiZAttention)
    assert injected.self_attn.mag_weight == 0.7
