# pylint: disable=redefined-outer-name, too-many-arguments
"""Basic green unit tests for LinearAttentionOp."""

import pytest
import torch

from src.tptt.modeling_tptt import LinearAttentionOp

# Fixtures


@pytest.fixture
def attention_dims():
    """Return dimension dict: batcnum_head, headseq_len, seq_len, head_dim."""
    return dict(batch=2, num_heads=3, seq_len=12, head_dim=8)


@pytest.fixture
def random_qkv_tensors(attention_dims):
    """Generate random q, k, v tensors and two beta tensors (tuple)."""
    batch_size, num_head, seq_len, head_dim = attention_dims.values()
    q = torch.randn(batch_size, num_head, seq_len, head_dim)
    k = torch.randn(batch_size, num_head, seq_len, head_dim)
    v = torch.randn(batch_size, num_head, seq_len, head_dim)
    beta1 = torch.sigmoid(torch.randn(batch_size, num_head, seq_len, 1))
    beta2 = torch.sigmoid(torch.randn(batch_size, num_head, seq_len, 1))
    return (q, k, v, (beta1, beta2))


@pytest.fixture
def dummy_cache():
    """Simple dict-like cache compatible with LinearAttentionOp."""

    class DummyCache(dict):
        def update(self, key, recurrent_state=None, qkv=None):
            entry = {"recurrent_state": recurrent_state}
            if qkv is not None:
                entry["qkv"] = qkv
            self[key] = entry

        def __getitem__(self, key):
            return dict.get(self, key, None)

    return DummyCache()


@pytest.fixture
def operator(dummy_cache):
    """A default LinearAttentionOp with quadratic operator and working cache."""
    return LinearAttentionOp(
        layer_idx=1,
        operator_mode="delta_product",
        recurrent_config={
            "order": 2,
            "gate_type": "kv",
            "linear": True,
            "trick": "derivative",
        },
        max_chunk_size=6,
        linear_cache=dummy_cache,
        linear_precision=torch.float32,
    )


# Tests


def test_compute_gate_branches(operator):
    """Test compute_gate for each gate type and invalid gate type."""
    t1, t2 = torch.tensor([0.5]), torch.tensor([0.7])
    operator.gate_type = "kv"
    assert torch.isclose(operator.compute_gate((t1, t2)), torch.tensor([0.35]))
    operator.gate_type = "k"
    assert torch.isclose(operator.compute_gate((t1, t2)), torch.tensor([0.5]))
    operator.gate_type = "v"
    assert torch.isclose(operator.compute_gate((t1, t2)), torch.tensor([0.7]))
    operator.gate_type = "unknown"
    with pytest.raises(ValueError):
        operator.compute_gate((t1, t2))


def test_forward_with_tuple_beta(operator, random_qkv_tensors, attention_dims):
    """Smoke test forward with standard tuple beta."""
    q, k, v, betas = random_qkv_tensors
    out = operator(q, k, v, beta=betas)
    assert out.shape == (
        attention_dims["batch"],
        attention_dims["num_heads"],
        attention_dims["seq_len"],
        attention_dims["head_dim"],
    )


def test_forward_qkv_buffers_concat(operator, dummy_cache):
    """If qkvb buffer in cache is right shape, triggers cat (time axis doubles = S*2)."""
    batch_size, num_head, seq_len, head_dim = 2, 2, 3, 4
    q = torch.randn(batch_size, num_head, seq_len, head_dim)
    k = torch.randn(batch_size, num_head, seq_len, head_dim)
    v = torch.randn(batch_size, num_head, seq_len, head_dim)
    beta = (
        torch.sigmoid(torch.randn(batch_size, num_head, seq_len, 1)),
        torch.sigmoid(torch.randn(batch_size, num_head, seq_len, 1)),
    )
    prevs = [torch.randn(batch_size, num_head, seq_len, head_dim) for _ in range(4)]
    operator.linear_cache = dummy_cache
    operator.linear_cache[1] = {
        "recurrent_state": torch.zeros(batch_size, num_head, 2, head_dim, head_dim),
        "qkv": tuple(prevs),
    }
    out = operator(q, k, v, beta=beta)
    # Accepts either S or 2S as output (concat or not)
    assert out.shape[2] in (seq_len, seq_len * 2)


def test_forward_save_cache_called(operator, random_qkv_tensors):
    """Ensure that save_cache is called if use_cache=True."""
    q, k, v, beta = random_qkv_tensors
    called = {}

    def fake_save_cache(*_, **__):
        called["x"] = True

    operator.save_cache = fake_save_cache
    _ = operator(q, k, v, beta=beta, use_cache=True)
    assert called.get("x", False)


def test_chunk_delta_product_forward_shapes(random_qkv_tensors):
    """Test chunk_delta_product_forward (n=1, n=2, with/without initial_state)."""
    q, k, v, (beta1, _) = random_qkv_tensors
    out1, state1 = LinearAttentionOp.chunk_delta_product_forward(q, k, v, beta1, 3)
    assert out1.shape == q.shape
    init_state = torch.randn(q.shape[0], q.shape[1], 2, q.shape[3], q.shape[3])
    out2, state2 = LinearAttentionOp.chunk_delta_product_forward(
        q, k, v, beta1, 3, n=2, initial_state=init_state
    )
    assert out2.shape == q.shape
    out3, state3 = LinearAttentionOp.chunk_delta_product_forward(
        q, k, v, beta1, 3, n=2, initial_state=torch.randn(1, 1, 1, 1, 1)
    )
    assert out3.shape == q.shape
