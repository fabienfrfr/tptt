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


def test_liza_attention_backward_with_mask(
    liza_attention, random_hidden_tensor, attention_mask
):
    """
    Test backward pass with attention mask.
    """
    random_hidden_tensor.requires_grad_(True)
    output, _ = liza_attention(random_hidden_tensor, attention_mask=attention_mask)
    loss = output.sum()
    loss.backward()
    for name, param in liza_attention.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in grad for {name}"


def test_liza_attention_backward_with_mask_multiple_steps(
    liza_attention, random_hidden_tensor, attention_mask, n_steps=3
):
    """
    Simulate multiple forward/backward passes with attention mask,
    in a production-like loop (optimizer, grad reset, train mode, etc.).
    """
    liza_attention.train()  # Ensure the module is in training mode
    optimizer = torch.optim.SGD(liza_attention.parameters(), lr=1e-3)

    for step in range(n_steps):
        optimizer.zero_grad()  # Reset gradients before each mini-batch

        # Simulate a new batch each step (clone/detach to avoid graph accumulation)
        input_tensor = random_hidden_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        output, _ = liza_attention(input_tensor, attention_mask=attention_mask)
        loss = output.sum()  # Replace with your actual loss function if needed

        # Backward pass
        loss.backward()
        optimizer.step()  # Update parameters as in real training

        # Check for NaNs in gradients after each backward
        for name, param in liza_attention.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(
                    param.grad
                ).any(), f"NaN in grad for {name} at step {step}"
