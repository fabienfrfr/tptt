import torch
import torch.nn as nn
from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
from fla.ops.delta_rule import fused_chunk_delta_rule, fused_recurrent_delta_rule

class FLAOperator:
    """Base class for FLA operators."""
    def __init__(self, mode="gla", head_dim=None):
        self.mode = mode
        self.head_dim = head_dim
        if mode == "gru" and head_dim is not None:
            self.gru = nn.GRU(head_dim, head_dim, batch_first=True)
        else:
            self.gru = None

    def __call__(self, q, k, v, g=None, scale=1.0, initial_state=None, training=True, **kwargs):
        if self.mode == "gla":
            return self._gla(q, k, v, g, scale, initial_state, training, **kwargs)
        elif self.mode == "delta_rule":
            return self._delta_rule(q, k, v, g, scale, initial_state, training, **kwargs)
        elif self.mode == "gru":
            return self._gru(q, v, initial_state)
        else:
            raise ValueError(f"Unknown FLA operator: {self.mode}")

    def _gla(self, q, k, v, g, scale, initial_state, training, **kwargs):
        if training or q.shape[-2] > 1:
            return fused_chunk_gla(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
        else:
            return fused_recurrent_gla(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)

    def _delta_rule(self, q, k, v, g, scale, initial_state, training, **kwargs):
        if training or q.shape[-2] > 1:
            return fused_chunk_delta_rule(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
        else:
            return fused_recurrent_delta_rule(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)

    def _gru(self, q, v, initial_state):
        # Merge heads for GRU, then split back
        b, h, n, d = q.shape
        x = v.permute(0,2,1,3).reshape(b, n, h*d)
        h0 = initial_state
        out, hn = self.gru(x, h0)
        out = out.reshape(b, n, h, d).permute(0,2,1,3)
        return out, hn

def get_fla_operator(name, head_dim=None):
    if isinstance(name, str):
        return FLAOperator(mode=name, head_dim=head_dim)
    elif callable(name):
        return name
    else:
        raise ValueError(f"Unknown FLA operator: {name}")
