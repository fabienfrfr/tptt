from liza.attention import ParallelFLAAttention
from liza.fla import gla_attention, delta_rule_attention
import torch

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

def test_gla_attention_shape():
    b, h, n, d = 2, 4, 10, 4
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    out = gla_attention(q, k, v)
    assert out.shape == (b, h, n, d)

def test_delta_rule_attention_shape():
    b, h, n, d = 2, 4, 10, 4
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    out = delta_rule_attention(q, k, v)
    assert out.shape == (b, h, n, d)
