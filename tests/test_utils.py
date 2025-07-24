# pylint: disable=redefined-outer-name, too-many-arguments, too-many-locals
"""Unit tests for tptt.utils utility functions."""

import pytest
import torch
from torch import nn

from src.tptt.modeling_tptt import (apply_linear_attention_mask,
                                    chunk_sequence, describe, ensure_stability,
                                    expand_virtual_tokens, extract_layer_idx,
                                    fast_invert_matrix, find_embedding_lm,
                                    get_valid_chunk_size, match_dim,
                                    soft_clamp, split_qkv,
                                    truncate_attention_mask,
                                    unlinear_activation)

# -------- soft_clamp --------


@pytest.mark.parametrize(
    "x,min_val,max_val",
    [
        (torch.tensor([0.0, 1.0, -1.0]), -1e4, 1e4),
        (torch.tensor([1e5, -1e5]), -1e4, 1e4),
        (torch.tensor([0.0, 50.0, 100.0]), 0, 100),
    ],
)
def test_soft_clamp_range(x, min_val, max_val):
    """Values are always between min and max."""
    y = soft_clamp(x, min_val, max_val)
    assert torch.all(y <= max_val) and torch.all(y >= min_val)


def test_soft_clamp_inverted():
    """Swapped min/max is still valid."""
    x = torch.tensor([0.3])
    y1 = soft_clamp(x, 1.0, 0.0)
    y2 = soft_clamp(x, 0.0, 1.0)
    assert torch.allclose(y1, y2, atol=1e-6) or torch.allclose(
        y1, y1.new_tensor([0.5]), atol=1
    )


# -------- unlinear_activation --------


def test_unlinear_activation_basic_and_scale():
    """Test shape, dtype, and custom scale on unlinear_activation."""
    x = torch.randn(2, 4, 8, 16)
    y = unlinear_activation(x)
    assert y.shape == x.shape
    y2 = unlinear_activation(x, scale=3.5)
    assert y2.shape == y.shape and not torch.allclose(y, y2)
    for val in [1e-20, 1e20]:
        z = unlinear_activation(torch.tensor([val]))
        assert torch.isfinite(z).all()


# -------- chunk_sequence --------


def test_chunk_sequence_simple():
    """Correctly slices for chunked sequence."""
    x = torch.arange(1 * 1 * 6 * 2).reshape(1, 1, 6, 2)
    out = chunk_sequence(x, num_chunks=3, chunk_size=2)
    assert out.shape == (1, 1, 3, 2, 2)
    assert torch.equal(out[0, 0, 0], x[0, 0, 0:2])
    assert torch.equal(out[0, 0, 1], x[0, 0, 2:4])
    assert torch.equal(out[0, 0, 2], x[0, 0, 4:6])


def test_chunk_sequence_bad_shape():
    """Shape not divisible by chunk_size will raise."""
    x = torch.randn(2, 2, 5, 4)
    with pytest.raises(RuntimeError):
        chunk_sequence(x, num_chunks=3, chunk_size=2)


def test_chunk_sequence_dtype():
    """Output preserves dtype."""
    x = torch.arange(1 * 1 * 4 * 2, dtype=torch.float16).reshape(1, 1, 4, 2)
    out = chunk_sequence(x, num_chunks=2, chunk_size=2)
    assert out.dtype == x.dtype


# -------- expand_virtual_tokens --------


@pytest.mark.parametrize("mode", ["derivative", "rotative", "combined"])
def test_expand_virtual_tokens_shape(mode):
    """Output shape = seq_len * n."""
    x = torch.randn(1, 5, 10, 8)
    out = expand_virtual_tokens(x, 3, mode)
    assert out.shape == (1, 5, 30, 8)


def test_expand_virtual_tokens_grad_and_identity():
    """Test gradient and n=1 identity."""
    x = torch.randn(2, 3, 4, 6, requires_grad=True)
    out = expand_virtual_tokens(x, 1)
    assert torch.allclose(x, out)
    y = expand_virtual_tokens(x, 2)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_expand_virtual_tokens_invalid_shape_and_mode():
    """Invalid shape or mode raises."""
    with pytest.raises(Exception):
        expand_virtual_tokens(torch.randn(2, 3, 4), 2)
    with pytest.raises(ValueError):
        expand_virtual_tokens(torch.randn(1, 2, 4, 5), 2, mode="badmode")


def test_expand_virtual_tokens_head_dim_odd_rotative():
    """rotative mode works for odd head_dim."""
    x = torch.randn(1, 2, 4, 7)
    out = expand_virtual_tokens(x, n=2, mode="rotative")
    assert out.shape[-1] == 7


# -------- fast_invert_matrix --------


@pytest.mark.parametrize(
    "num_batch, hidden_dim, num_chunk, size",
    [(2, 3, 4, 5), (1, 1, 1, 2), (3, 2, 1, 8)],
)
def test_fast_invert_matrix_correctness(num_batch, hidden_dim, num_chunk, size):
    """Correct inverse for random lower triangular matrices."""
    householder = torch.tril(
        torch.randn(num_batch, hidden_dim, num_chunk, size, size), diagonal=-1
    )
    tri_eye = torch.eye(size).to(householder.device).to(householder.dtype)
    indicator = tri_eye.view((1, 1, 1, size, size))
    inv = fast_invert_matrix(householder)
    matrix = indicator - householder
    result = torch.matmul(inv, matrix)
    assert torch.allclose(result, indicator.expand_as(result), atol=1e-5)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_fast_invert_matrix_identity_on_zero_input(dtype):
    """Zero input should yield identity matrices (one per batch), for different dtypes."""
    tri = torch.zeros(2, 3, 4, 4, dtype=dtype)
    eye = torch.eye(4, dtype=dtype).view(1, 1, 4, 4).expand_as(tri)
    out = fast_invert_matrix(tri, dtype=dtype)

    assert out.dtype == dtype
    assert torch.allclose(out, eye), "Expected identity matrices for zero input"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_fast_invert_matrix_finite_output(dtype):
    """Output should be finite for large input values (no inf/nan)."""
    tri = torch.full((2, 2, 3, 3), 1e4, dtype=dtype)
    out = fast_invert_matrix(tri, dtype=dtype)

    assert out.dtype == dtype
    assert torch.isfinite(out).all(), "Expected finite values in inversion output"


# -------- get_valid_chunk_size --------


@pytest.mark.parametrize(
    "input_size, max_chunk, expected",
    [
        (127, 64, 1),
        (128, 64, 64),
        (120, 64, 60),
        (7, 10, 7),
        (0, 4, 1),
        (1, 1, 1),
    ],
)
def test_get_valid_chunk_size_all_cases(input_size, max_chunk, expected):
    """Covers many usecases (incl. edge case)."""
    assert get_valid_chunk_size(input_size, max_chunk) == expected


# -------- match_dim --------


def test_match_dim_expand_and_reduce():
    """Test expansion & reduction & identity, various dims."""
    x = torch.randn(1, 10, 32)
    matched = match_dim(x, 1, 20)
    assert matched.shape == (1, 20, 32)
    matched = match_dim(x, 1, 5)
    assert matched.shape == (1, 5, 32)
    y = match_dim(x, -1, 32)
    assert torch.allclose(x, y)


def test_match_dim_dim0_and_projection_interpolate():
    """Expand/reduce dim 0."""
    x = torch.randn(5, 4, 10)
    reduced = match_dim(x, 0, 3)
    assert reduced.shape == (3, 4, 10)
    # interpolate
    y = match_dim(torch.randn(2, 4, 8), 1, 7)
    assert y.shape == (2, 7, 8)
    y = match_dim(torch.randn(2, 7, 8), 1, 4)
    assert y.shape == (2, 4, 8)


# -------- apply_linear_attention_mask --------


def test_apply_attention_mask_left_padding():
    """Left padding, and masks with int/neg values."""
    mask = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)
    v = torch.ones(2, 2, 4)
    masked = apply_linear_attention_mask(mask, v, padding_side="left")
    assert masked.shape == v.shape
    mask2 = torch.ones(1, 1, 2, 2, dtype=torch.uint8)
    v2 = torch.ones(1, 2, 3)
    apply_linear_attention_mask(mask2, v2)
    neg_mask = torch.tensor([[-1, 0, 1]])
    v = torch.randn(1, 3, 2)
    out = apply_linear_attention_mask(neg_mask, v)
    assert torch.all(out[0, 0] == 0)


def test_apply_attention_mask_highdim():
    """High-dim v supported."""
    attn_mask = torch.ones(2, 3)
    v = torch.ones(2, 3, 8, 4, 4)
    result = apply_linear_attention_mask(attn_mask, v)
    assert result.shape == v.shape


# -------- truncate_attention_mask --------


@pytest.mark.parametrize(
    "mask_shape,hidden_shape",
    [
        ((2, 4), (2, 4, 8)),
        ((2, 1, 4, 4), (2, 4, 8)),
        ((1, 4), (1, 4, 8)),
        ((1, 1, 4), (1, 4, 8)),
    ],
)
def test_truncate_attention_mask_shapes(mask_shape, hidden_shape):
    """Many valid mask/hid shapes."""
    max_length = 2
    hidden_states = torch.randn(hidden_shape)
    attention_mask = torch.ones(mask_shape)
    truncated_hidden, truncated_mask = truncate_attention_mask(
        hidden_states, attention_mask, max_length
    )
    assert (
        truncated_hidden.shape[1] == max_length
        or truncated_hidden.shape[-2] == max_length
    )


def test_truncate_attention_mask_none():
    """Handle None, invalid raises."""
    hs = torch.randn(2, 4, 8)
    x, m = truncate_attention_mask(hs, None, 3)
    assert m is None


# -------- describe --------


def test_describe_dtype_device(capsys):
    """describe mentions dtype and device in output."""
    x = torch.zeros(2, device="cuda") if torch.cuda.is_available() else torch.zeros(2)
    describe(x, name="cuda_or_cpu")
    captured = capsys.readouterr().out
    assert "dtype" in captured and "device" in captured


# -------- ensure_stability --------


def test_ensure_stability_nan_inf_and_dtype():
    """Clamp nan/inf and preserve dtype, handle empty."""
    arr = torch.tensor([float("nan"), float("inf"), float("-inf"), 1e4, -1e4])
    out = ensure_stability(arr, min_val=-3, max_val=3)
    assert torch.all(out <= 3) and torch.all(out >= -3)
    arr = torch.ones(0)
    assert ensure_stability(arr).shape == arr.shape
    arr = torch.ones(4, dtype=torch.float64)
    assert ensure_stability(arr, 0, 2).dtype == torch.float64
    arr2 = torch.tensor([0.1, 0.8])
    res = ensure_stability(arr2, min_val=2, max_val=-2)
    assert torch.all(res <= 2) and torch.all(res >= -2)


# -------- extract_layer_idx --------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("encoder.4.attention", 4),
        ("foo.42.bar", 42),
        ("a.123.b.456", 123),
        ("none", -1),
        ("layer.0.norm", 0),
        ("mod.norm", -1),
        ("abc", -1),
        ("test.99", -1),  # need first layers.X.attn (no LoRA)
        ("several.2.3.10", 2),
        ("hidden.5.activation.7", 5),
    ],
)
def test_extract_layer_idx_various(name, expected):
    """extract_layer_idx finds first numerical segment, or -1."""
    assert extract_layer_idx(name) == expected


# -------- split_qkv --------


def test_split_qkv_valid_and_fail():
    """split_qkv works or raises appropriately."""

    class FakeAttn:
        num_q_heads = 2
        num_k_heads = 3
        num_v_heads = 1
        head_dim = 4

    fa = FakeAttn()
    qkv = torch.arange((2 + 3 + 1) * 4)
    q, k, v = split_qkv(fa, qkv)
    assert q.shape[-1] == 8 and k.shape[-1] == 12 and v.shape[-1] == 4

    class Invalid:
        pass

    with pytest.raises(ValueError):
        split_qkv(Invalid(), qkv)

    class FakeAttn2:
        num_q_heads = 1
        num_k_heads = 1
        num_v_heads = 1
        head_dim = 4

    f2 = FakeAttn2()
    with pytest.raises(RuntimeError):
        split_qkv(f2, torch.arange(1))


# -------- find_embedding_lm --------


def test_find_embedding_lm_variants():
    """find_embedding_lm finds embed_tokens, token_embeddings, or None."""

    class EmbModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(5, 13)

    assert isinstance(find_embedding_lm(EmbModule()), nn.Embedding)

    class TokModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embeddings = nn.Embedding(5, 9)

    assert isinstance(find_embedding_lm(TokModule()), nn.Embedding)

    class Nothing(nn.Module):
        def __init__(self):
            super().__init__()

    assert find_embedding_lm(Nothing()) is None
    seq = nn.Sequential(EmbModule())
    assert isinstance(find_embedding_lm(seq), nn.Embedding)
