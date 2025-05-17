# pylint: disable=redefined-outer-name
"""Tests for LiZAttention projection sharing."""

from src.tptt.liza.memory_gate import LiZAttention


def test_projection_sharing(dummy_base_attn, dummy_config):
    """Test that LiZAttention shares projections with the base attention module."""
    lin_attn = LiZAttention(dummy_base_attn, 1, dummy_config)
    assert lin_attn.q_proj is dummy_base_attn.q_proj
    assert lin_attn.o_proj is dummy_base_attn.o_proj
