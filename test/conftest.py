import pytest
import torch
import torch.nn as nn
from liza.config import AttentionConfig

class DummyBaseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.o_proj = nn.Linear(16, 16)
        self.config = AttentionConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=16
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Return dummy output and dummy attention weights
        return hidden_states, None

@pytest.fixture
def dummy_base_attention():
    return DummyBaseAttention()

@pytest.fixture
def dummy_input():
    return torch.randn(2, 10, 16)
