import pytest
import torch.nn as nn

from liza.injector import inject_linear_attention
from liza.linear_attention import LinearAttention


class DummySelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)

    def forward(self, x, **kwargs):
        # Simule la sortie d'un module d'attention standard
        return torch.randn(x.shape[0], x.shape[1], 64), None


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttn()


class DummyConfig:
    hidden_size = 64
    num_attention_heads = 8
    num_key_value_heads = 4
    attention_dropout = 0.1
    head_dim = 8


def test_inject_linear_attention_replaces_module():
    model = DummyModel()
    injected = inject_linear_attention(
        model, DummyConfig(), target_modules=["self_attn"], fla_weight=0.7
    )
    assert isinstance(injected.self_attn, LinearAttention)
    assert injected.self_attn.fla_weight == 0.7
