# pylint: disable=redefined-outer-name
"""Tests for LiZAttention projection sharing."""


def test_projection_base_sharing(dummy_base_attn, liza_attention):
    """Test that LiZAttention shares projections with the base attention module."""
    lin_attn = liza_attention
    assert lin_attn.base_attn is dummy_base_attn


def test_attention_output_shape(
    liza_attention, random_hidden_tensor, batch_size, seq_len, tensor_dim
):
    """Test the output shape of the attention module."""
    output, _ = liza_attention(random_hidden_tensor)
    assert output.shape == (
        batch_size,
        seq_len,
        tensor_dim,
    )
