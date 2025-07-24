"""Tests for the extra linear attention module."""

import torch


def test_linear_attention_forward(linear_attention):
    """Test linear attention only forward (contain ops calculus)"""
    num_batch, seq_len, hidden_dim = 2, 16, 32  # batch, seq_len, hidden_dim

    x = torch.randn(num_batch, seq_len, hidden_dim)

    y = linear_attention(x)
    assert y.shape == (num_batch, seq_len, hidden_dim)
    print("Test passed: output shape is", y.shape)


def test_bidirectional_linear_attention_forward(bidirectional_linear_attention):
    """Test bidirectional linear attention with ops"""
    num_batch, seq_len, hidden_dim = 2, 16, 32  # batch, seq_len, hidden_dim

    x = torch.randn(num_batch, seq_len, hidden_dim)

    y = bidirectional_linear_attention(x)
    assert y.shape == (num_batch, seq_len, hidden_dim)
    print("Test passed: output shape is", y.shape)
