"""Tests for the extra linear attention module."""

import torch


def test_linear_attention_forward(linear_attention):
    B, S, D = 2, 16, 32  # batch, seq_len, hidden_dim

    x = torch.randn(B, S, D)

    y = linear_attention(x)
    assert y.shape == (B, S, D)
    print("Test passed: output shape is", y.shape)
