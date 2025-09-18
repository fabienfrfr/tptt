# pylint: disable=redefined-outer-name, too-many-arguments
"""Basic green unit tests for LinearAttentionOp."""

import pytest
import torch

from src.tptt.modeling_tptt import LinearAttentionOp


def test_linear_operator_forward_output(
    linear_operator, random_qkv_expended_tensors, attention_dims
):
    """Smoke test forward with standard tuple beta."""
    q, k, v, alpha, beta = random_qkv_expended_tensors
    n_orders = q.shape[3]
    linear_activation = True
    out, state = linear_operator(q, k, v, alpha, beta, linear_activation)
    assert out.shape == (
        attention_dims["batch"],
        attention_dims["num_heads"],
        attention_dims["seq_len"],
        attention_dims["head_dim"],
    )
    assert state.shape == (
        attention_dims["batch"],
        attention_dims["num_heads"],
        attention_dims["head_dim"],
        attention_dims["head_dim"],
    )


@pytest.mark.parametrize(
    "initial_state",
    [
        None,
        pytest.param(torch.randn(1, 1, 1, 1, 1), id="dummy_state"),
    ],
)
def test_chunk_delta_product_forward_shapes(random_qkv_expended_tensors, initial_state):
    """Test chunk_delta_product_forward (n=1, n=2, with/without initial_state)."""
    q, k, v, alpha, beta = random_qkv_expended_tensors
    batch_size, num_heads, seq_len, _, head_dim = q.shape
    kwargs = {"initial_state": initial_state} if initial_state is not None else {}
    out, _ = LinearAttentionOp.chunk_delta_product_forward(
        q, k, v, alpha, beta, chunk_size=3, **kwargs
    )
    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
