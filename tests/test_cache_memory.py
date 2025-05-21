"""Unit tests for the Cache utility class."""

import torch


def test_cache_initial_state(cache):
    """Test that cache initializes with empty state and zero tokens."""
    assert cache.states == []
    assert cache.seen_tokens == 0


def test_cache_update_and_getitem(cache, batch_size, seq_len, head_dim):
    """Test cache update and retrieval for one layer."""
    tensor = torch.randn(batch_size, seq_len, head_dim)
    cache.update(0, key=tensor)
    state = cache[0]
    assert isinstance(state, dict)
    assert torch.equal(state["key"], tensor)


def test_cache_update_multiple_layers(cache, batch_size, seq_len, head_dim):
    """Test cache update for multiple layers."""
    t1 = torch.randn(batch_size, seq_len, head_dim)
    t2 = torch.randn(batch_size, seq_len, head_dim)
    cache.update(0, key=t1)
    cache.update(1, key=t2)
    assert torch.equal(cache[0]["key"], t1)
    assert torch.equal(cache[1]["key"], t2)


def test_cache_update_overwrite(cache, batch_size, seq_len, head_dim):
    """Test that cache update overwrites previous value for a layer."""
    t1 = torch.randn(batch_size, seq_len, head_dim)
    t2 = torch.randn(batch_size, seq_len, head_dim)
    cache.update(0, key=t1)
    cache.update(0, key=t2)
    assert torch.equal(cache[0]["key"], t2)


def test_cache_sliding_window_not_applied_for_1d(
    cache_with_max_length, batch_size, max_length
):
    """Test that sliding window is not applied for 1D tensors."""
    t = torch.arange(batch_size * (max_length + 2), dtype=torch.float32)
    cache_with_max_length.update(0, key=t)
    state = cache_with_max_length[0]
    assert torch.equal(state["key"], t)


def test_cache_reset(cache, batch_size, seq_len, head_dim):
    """Test cache reset clears state and token count."""
    t = torch.randn(batch_size, seq_len, head_dim)
    cache.update(0, key=t)
    cache.seen_tokens = 42
    cache.reset()
    assert cache.states == []
    assert cache.seen_tokens == 0


def test_cache_get_max_length(cache_with_max_length, max_length):
    """Test get_max_length returns the correct value."""
    assert cache_with_max_length.get_max_length() == max_length


def test_cache_update_with_multiple_keys(cache, batch_size, seq_len, head_dim):
    """Test cache update with multiple keys in one call."""
    t1 = torch.randn(batch_size, seq_len, head_dim)
    t2 = torch.randn(batch_size, seq_len, head_dim)
    cache.update(0, key1=t1, key2=t2)
    state = cache[0]
    assert torch.equal(state["key1"], t1)
    assert torch.equal(state["key2"], t2)


def test_cache_getitem_out_of_bounds(cache):
    """Test that accessing out-of-bounds cache returns None."""
    assert cache[10] is None
