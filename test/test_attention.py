import torch
from liza.attention import ParallelFLAAttention
from liza.fla import get_fla_operator

from unittest.mock import patch

@patch("fla.ops.gla.fused_chunk_gla", return_value="mocked_result")
@patch("fla.ops.gla.fused_recurrent_gla", return_value="mocked_result")
def test_fla_operator(mock_fused_chunk_gla, mock_fused_recurrent_gla):
    from liza.fla import FLAOperator
    operator = FLAOperator(mode="gla")
    result = operator(q=None, k=None, v=None)
    assert result == "mocked_result"

def test_parallel_fla_attention_gla(dummy_base_attention, dummy_input):
    config = dummy_base_attention.config
    module = ParallelFLAAttention(dummy_base_attention, config, operator="gla")
    out, attn = module(dummy_input)
    assert out.shape == dummy_input.shape
    assert attn is None

def test_parallel_fla_attention_delta_rule(dummy_base_attention, dummy_input):
    config = dummy_base_attention.config
    module = ParallelFLAAttention(dummy_base_attention, config, operator="delta_rule")
    out, attn = module(dummy_input)
    assert out.shape == dummy_input.shape
    assert attn is None

def test_parallel_fla_attention_gru(dummy_base_attention, dummy_input):
    config = dummy_base_attention.config
    module = ParallelFLAAttention(dummy_base_attention, config, operator="gru")
    out, attn = module(dummy_input)
    assert out.shape == dummy_input.shape
    assert attn is None

def test_gla_operator_shape():
    b, h, n, d = 2, 4, 10, 4
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    g = torch.zeros_like(q)
    operator = get_fla_operator("gla", head_dim=d)
    out, _ = operator(q, k, v, g, scale=1.0, training=True)
    assert out.shape == (b, h, n, d)

def test_delta_rule_operator_shape():
    b, h, n, d = 2, 4, 10, 4
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    g = torch.zeros_like(q)
    operator = get_fla_operator("delta_rule", head_dim=d)
    out, _ = operator(q, k, v, g, scale=1.0, training=True)
    assert out.shape == (b, h, n, d)

def test_gru_operator_shape():
    b, h, n, d = 2, 4, 10, 4
    q = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    operator = get_fla_operator("gru", head_dim=d)
    out, _ = operator(q, None, v, None, scale=1.0, training=True)
    assert out.shape == (b, h, n, d)
