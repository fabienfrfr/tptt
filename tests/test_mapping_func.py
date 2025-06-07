# pylint: disable=redefined-outer-name
"""Tests for the AttentionOperator module."""

import sys
import types
from unittest import mock

import pytest

from src.tptt.modeling_tptt import AttentionOperator, import_fla_ops


def test_forward_shape(
    operator,
    random_qkv_tensors,
    chunk_size,
    seq_len,
    batch_size,
    num_heads,
    head_dim,
):
    """Test output shape of the forward pass."""
    q, k, v, beta = random_qkv_tensors
    o, _ = operator(q, k, v, beta=beta, chunk_size=chunk_size)
    assert o.shape == (batch_size, num_heads, seq_len, head_dim)


def test_attention_operator_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        op = AttentionOperator(mode="not_a_mode")
        op(None, None, None)


def test_chunk_delta_rule_forward_computation(
    random_qkv_tensors, batch_size, num_heads, seq_len, head_dim
):
    q, k, v, beta = random_qkv_tensors
    chunk_size = 8
    out, _ = AttentionOperator.chunk_delta_rule_forward(q, k, v, beta, chunk_size)
    assert out.shape == (batch_size, num_heads, seq_len, head_dim)


@pytest.mark.parametrize(
    "cuda_available,import_raises,expected",
    [
        (True, False, True),  # CUDA dispo
        (False, False, None),  # CUDA non dispo
    ],
)
def test_import_gla_ops(cuda_available, import_raises, expected):
    with mock.patch("torch.cuda.is_available", return_value=cuda_available):
        if cuda_available and not import_raises:
            gla_mod = types.ModuleType("fla.ops.gla")
            gla_mod.fused_chunk_gla = lambda: None
            gla_mod.fused_recurrent_gla = lambda: None
            sys.modules["fla.ops.gla"] = gla_mod
        elif cuda_available and import_raises:
            if "fla.ops.gla" in sys.modules:
                del sys.modules["fla.ops.gla"]

        result = import_fla_ops()

        if expected is True:
            assert callable(result[0]) and callable(result[1])
        elif expected is False:
            assert result == (None, None)
        else:
            assert result == (None, None)

        if "fla.ops.gla" in sys.modules:
            del sys.modules["fla.ops.gla"]
