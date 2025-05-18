"""Unit test for inject_linear_attention replacing a module with LiZAttention."""

from src.tptt.injection import inject_linear_attention
from src.tptt.liza.memory_gate import LiZAttention


def test_inject_linear_attention_replaces_module(dummy_decoder, dummy_config):
    """Test that inject_linear_attention replaces the target module with LiZAttention."""
    model = dummy_decoder()
    injected = inject_linear_attention(
        model, dummy_config, target_modules=["self_attn"], mag_weight=0.7
    )
    assert isinstance(injected.self_attn, LiZAttention)
    assert injected.self_attn.mag_weight == 0.7
