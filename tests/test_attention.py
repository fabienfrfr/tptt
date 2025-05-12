import pytest
import torch

from liza.mpa import ParallelFLAAttention


@pytest.mark.parametrize("operator", ["delta_rule", "gla"])
@pytest.mark.parametrize("use_mask", [True, False])
def test_parallel_fla_attention_forward(operator, use_mask, base_attn, config):
    B, N = 2, 5
    hidden_states = torch.randn(B, N, config.hidden_size)
    attention_mask = torch.ones(B, N, 1) if use_mask else None

    fla = ParallelFLAAttention(
        base_attn=base_attn,
        config=config,
        operator=operator,
        fla_weight=0.5,
    )

    out, attn_weights = fla(hidden_states, attention_mask=attention_mask)
    assert out.shape == (B, N, config.hidden_size)
    assert attn_weights.shape[0] == B

def test_parallel_fla_attention_cache(base_attn, config):
    B, N = 2, 5
    hidden_states = torch.randn(B, N, config.hidden_size)
    attention_mask = torch.ones(B, N, 1)
    layer_idx = 0

    fla = ParallelFLAAttention(
        base_attn=base_attn,
        config=config,
        operator="delta_rule",
        fla_weight=0.5,
        layer_idx=layer_idx,
    )

    past_key_value = {layer_idx: torch.zeros(B, config.num_attention_heads, config.head_dim)}
    out, attn_weights = fla(hidden_states, attention_mask=attention_mask, past_key_value=past_key_value)
    assert out.shape == (B, N, config.hidden_size)
    assert isinstance(attn_weights, torch.Tensor)

def test_parallel_fla_attention_backward(base_attn, config):
    B, N = 2, 5
    hidden_states = torch.randn(B, N, config.hidden_size, requires_grad=True)
    attention_mask = torch.ones(B, N, 1)

    fla = ParallelFLAAttention(
        base_attn=base_attn,
        config=config,
        operator="delta_rule",
        fla_weight=0.5,
    )
    out, _ = fla(hidden_states, attention_mask=attention_mask)
    loss = out.sum()
    loss.backward()
    assert hidden_states.grad is not None
    assert hidden_states.grad.shape == hidden_states.shape
