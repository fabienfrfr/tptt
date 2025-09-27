"""Tests for the extra linear attention module."""

import torch
import pytest


def test_output_shape__causal_conv1d(causal_conv1d):
    # Input tensor: [batch=2, channels=1, seq_len=8]
    x = torch.randn(2, 1, 8)
    y = causal_conv1d(x)
    # Output should match input shape except for batch and channel settings
    assert y.shape == (2, 1, 8)


def test_expected_values_causal_conv1d(causal_conv1d):
    """
    Test that causal 1D convolution (offset=0, replicate padding) matches manual windowed average.

    Input sequence is [0, 1, 2, 3, 4, 5] (batch=1, channel=1). Kernel size=3, all weights=1/3.

    Expected results are as follows:
      Position 0: mean(0,0,0) = (0+0+0)/3 = 0.0
      Position 1: mean(0,0,1) = (0+0+1)/3 = 0.333...
      Position 2: mean(0,1,2) = 1
      Position 3: mean(1,2,3) = 2
      Position 4: mean(2,3,4) = 3
      Position 5: mean(3,4,5) = 4

    Asserts values are equal up to tolerance accounting for float precision.
    """
    # Input tensor, shape [B=1, C=1, S=6]
    x = torch.arange(0, 6, dtype=torch.float32).view(1, 1, 6)
    y = causal_conv1d(x)  # Output: [1, 1, 6]

    expected = torch.tensor([0, 0.3333333, 1, 2, 3, 4], dtype=torch.float32).view(
        1, 1, 6
    )
    assert torch.allclose(y, expected, atol=1e-5)


def test_avgpool_output_shape(causal_avgpool):
    x = torch.randn(2, 8, 16)  # [B, S, F]
    y = causal_avgpool(x)
    # Expected shape: [B, S, output_size]
    assert y.shape == (2, 8, 4)


def test_avgpool_expected_values(causal_avgpool):
    """
    Verify that the causal sliding average matches manual expected values for a simple increasing sequence.
    """
    # Create a simple increasing sequence repeated on 16 features to match input_size
    base_seq = torch.arange(0, 6, dtype=torch.float32).view(
        1, 6, 1
    )  # Shape: [batch=1, seq_len=6, features=1]
    x = base_seq.repeat(1, 1, 16)  # Shape: [1, 6, 16]

    # Run the causal_avgpool forward pass
    y = causal_avgpool(x)  # Expected output shape: [1, 6, 4]

    # Manually computed expected causal averages for the sequence with replicate padding
    expected_single_feature = torch.tensor(
        [0, 1 / 3, 1, 2, 3, 4], dtype=torch.float32
    ).view(1, 6, 1)

    # Repeat the expected values on the output feature dimension (4) for comparison
    expected_expanded = expected_single_feature.repeat(1, 1, 4)

    # Assert the model output matches the expected values within tolerance
    assert torch.allclose(
        y, expected_expanded, atol=1e-5
    ), f"Output differs from expected:\n{y}\nExpected:\n{expected_expanded}"


def test_deltarule_attention_forward(deltarule_attention):
    """Test deltarule attention only forward (contain ops calculus)"""
    num_batch, seq_len, hidden_dim = 2, 16, 32  # batch, seq_len, hidden_dim

    x = torch.randn(num_batch, seq_len, hidden_dim)

    y = deltarule_attention(x)
    assert y.shape == (num_batch, seq_len, hidden_dim)
    print("Test passed: output shape is", y.shape)


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


def test_compute_gate_branches(linear_attention):
    # Test valid and invalid gate types for gate computation
    channels = linear_attention.hidden_dim  # head_dim * num_key_value_heads
    t1 = torch.randn(2, 4, channels)
    t2 = torch.randn(2, 4, channels)

    for gate in ["kv", "k", "v", "c"]:
        linear_attention.recurrent_config["alpha_gate"] = gate
        linear_attention.recurrent_config["beta_gate"] = gate
        alpha, beta = linear_attention.compute_gate(t1, t2)
        # Check shape and finite output
        assert alpha.shape == beta.shape
        assert torch.isfinite(alpha).all()
        assert torch.isfinite(beta).all()

    # Test invalid gate type handling (should raise KeyError)
    linear_attention.recurrent_config["alpha_gate"] = "unknown"
    linear_attention.recurrent_config["beta_gate"] = "unknown"
    with pytest.raises(KeyError):
        linear_attention.compute_gate(t1, t2)


def test_forward_qkv_buffers_concat(linear_attention, dummy_cache):
    # Ensure cache concatenation works if buffer shape matches
    batch_size, seq_len, hidden_dim = 2, 4, linear_attention.hidden_dim
    x = torch.randn(batch_size, seq_len, hidden_dim)
    prevs = [torch.randn_like(x) for _ in range(3)] + [
        torch.randn_like(x),
        torch.randn_like(x),
    ]
    linear_attention.linear_cache = dummy_cache
    linear_attention.layer_idx = 1
    linear_attention.linear_cache[1] = {
        "recurrent_state": torch.zeros(
            batch_size,
            linear_attention.num_heads,
            2,
            linear_attention.head_dim,
            linear_attention.head_dim,
        ),
        "qkvg": tuple(prevs),
    }
    out = linear_attention(x, use_cache=True)
    # Output seq_len is either seq_len or 2*seq_len after concat
    assert out.shape[1] in (seq_len, seq_len * 2)


def test_forward_save_cache_called(linear_attention):
    # Verify save_cache is called when use_cache=True
    batch_size, seq_len, hidden_dim = 2, 4, linear_attention.hidden_dim
    x = torch.randn(batch_size, seq_len, hidden_dim)
    called = {}

    def fake_save_cache(q, k, v, alpha, beta, state, n_orders):
        called["x"] = True

    linear_attention.save_cache = fake_save_cache
    _ = linear_attention(x, use_cache=True)
    assert called.get("x", False)


def test_forward_save_cache_shape(linear_attention):
    # Verify save_cache stores correct tensor shapes
    batch_size, seq_len, hidden_dim = 2, 4, linear_attention.hidden_dim
    x = torch.randn(batch_size, seq_len, hidden_dim)
    saved = {}

    def fake_save_cache(q, k, v, alpha, beta, state, n_orders):
        saved["q"] = q

    linear_attention.save_cache = fake_save_cache

    _ = linear_attention(x, use_cache=True)
    assert saved["q"].shape == (
        batch_size,
        linear_attention.num_heads,
        seq_len,
        linear_attention.head_dim,
    )


def test_forward_save_cache_values(linear_attention):
    # Verify save_cache stores correct tensor shapes
    batch_size, seq_len, hidden_dim = 2, 4, linear_attention.hidden_dim
    x = torch.randn(batch_size, seq_len, hidden_dim)
    saved = {}

    def fake_save_cache(q, k, v, alpha, beta, state, n_orders):
        saved["q"] = q
        saved["k"] = k
        saved["v"] = v

    linear_attention.save_cache = fake_save_cache

    # manually prepare q, k, v to check values
    q = linear_attention.q_proj(x)
    k = linear_attention.k_proj(x)
    v = linear_attention.v_proj(x)
    q, k, v = linear_attention.prepare_attention_input(q, k, v)

    _ = linear_attention(x, use_cache=True)
    assert torch.allclose(saved["q"], q)
    assert torch.allclose(saved["k"], k)
    assert torch.allclose(saved["v"], v)
