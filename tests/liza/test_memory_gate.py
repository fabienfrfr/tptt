# pylint: disable=redefined-outer-name
"""Tests for LiZAttention projection sharing."""
import torch


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


def test_attention_output_shape_with_mask(
    liza_attention,
    random_hidden_tensor,
    batch_size,
    seq_len,
    tensor_dim,
    attention_mask,
):
    """Test the output shape of the attention module with an attention mask."""
    # Create a mask of shape [batch_size, seq_len] filled with 1s (no masking)
    output, _ = liza_attention(random_hidden_tensor, attention_mask=attention_mask)
    assert output.shape == (batch_size, seq_len, tensor_dim)


def test_liza_attention_fused_qkv_with_mask(
    batch_size, seq_len, tensor_dim, liza_qkv_attention, attention_mask
):
    """Test LiZAttention with a fused QKV projection dummy module."""
    x = torch.randn(batch_size, seq_len, tensor_dim)
    output, _ = liza_qkv_attention(x, attention_mask)

    # Output shape should be (batch, seq_len, tensor_dim)
    assert output.shape == (batch_size, seq_len, tensor_dim)
