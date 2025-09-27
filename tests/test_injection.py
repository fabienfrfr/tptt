"""Unit test for inject_linear_attention replacing a module with LiZAttention."""

from src.tptt.modeling_tptt import LiZAttention, get_tptt_model


def test_inject_linear_attention_replaces_module(dummy_decoder, dummy_config):
    """Test that inject_linear_attention replaces the target module with LiZAttention."""
    model = dummy_decoder()
    injected, _ = get_tptt_model(
        model,
        dummy_config,
        liza_attention=LiZAttention,
        target_modules_names=["self_attn"],
        mag_weight=0.7,
    )
    assert isinstance(injected.self_attn, LiZAttention)
    assert injected.self_attn.memory_gate.mag_weight == 0.7
