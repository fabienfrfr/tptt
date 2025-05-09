import torch
from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
from fla.ops.delta_rule import fused_chunk_delta_rule, fused_recurrent_delta_rule


def gla_attention(q, k, v, g=None, scale=1.0, **kwargs):
    """
    Flash Linear Attention (GLA) wrapper.
    """
    if g is None:
        g = torch.zeros_like(q)
    out, _ = fused_chunk_gla(q, k, v, g, scale=scale, initial_state=None, output_final_state=False)
    return out

def delta_rule_attention(q, k, v, g=None, scale=1.0, **kwargs):
    """
    Delta Rule Attention wrapper.
    """
    if g is None:
        g = torch.zeros_like(q)
    out, _ = fused_chunk_delta_rule(q, k, v, g, scale=scale, initial_state=None, output_final_state=False)
    return out
