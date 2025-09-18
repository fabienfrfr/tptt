"""Tests for the extra linear attention module."""

import torch
import pytest


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
    t1 = torch.randn(2, 4, linear_attention.head_dim)
    t2 = torch.randn(2, 4, linear_attention.head_dim)

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
