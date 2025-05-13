import pytest
import torch
import torch.nn as nn

from liza.linear_attention import LinearAttention


@pytest.fixture
def dummy_base_attn():
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)

        def forward(self, x, **kwargs):
            # Simule la sortie d'un module d'attention standard
            return torch.randn(x.shape[0], x.shape[1], 64), None

    return DummyAttention()


@pytest.fixture
def dummy_config():
    class DummyConfig:
        hidden_size = 64
        num_attention_heads = 8
        num_key_value_heads = 4
        attention_dropout = 0.1
        head_dim = 8

    return DummyConfig()


def test_projection_sharing(dummy_base_attn, dummy_config):
    lin_attn = LinearAttention(dummy_base_attn, dummy_config)
    assert lin_attn.q_proj is dummy_base_attn.q_proj
    assert lin_attn.o_proj is dummy_base_attn.o_proj
