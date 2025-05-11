import pytest
import torch
import torch.nn as nn

from liza.config import AttentionConfig

@pytest.fixture
def config():
    class DummyConfig:
        hidden_size = 16
        num_attention_heads = 4
        num_key_value_heads = 2
        head_dim = 4
    return DummyConfig()

@pytest.fixture
def base_attn(config):
    class DummyBaseAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.config = config

        def forward(self, hidden_states, **kwargs):
            out = self.o_proj(hidden_states)
            attn_weights = torch.ones(hidden_states.shape[0], hidden_states.shape[1])
            return out, attn_weights

    return DummyBaseAttention(config)