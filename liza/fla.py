import torch
import torch.nn as nn

def check_triton_availability():
    try:
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU detected. Triton requires a GPU.")
        print("Triton is available.")
        return True
    except (ImportError, RuntimeError) as e:
        import warnings
        warnings.warn(f"Triton is not available: {e}. Falling back to PyTorch.")
        return False

TRITON_AVAILABLE = check_triton_availability()

if TRITON_AVAILABLE:
    from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
    from fla.ops.delta_rule import fused_chunk_delta_rule, fused_recurrent_delta_rule

def gla_recurrent(q, k, v, g=None, scale=1.0):
    # q, k, v: [B, H, N, D], g: [B, H, N, 1]
    B, H, N, D = q.shape
    if g is None:
        lambd = torch.zeros(B, H, N, 1, dtype=q.dtype, device=q.device)
    else:
        lambd = torch.sigmoid(g)
    q = q * scale
    S = torch.zeros(B, H, D, D, dtype=q.dtype, device=q.device)
    outs = []
    for t in range(N):
        kvT = torch.einsum('bhd,bhe->bhde', k[:, :, t, :], v[:, :, t, :])
        lmbd = lambd[:, :, t, :].reshape(B, H, 1, 1)
        S = lmbd * S + (1 - lmbd) * kvT
        o = torch.einsum('bhd,bhde->bhe', q[:, :, t, :], S)
        outs.append(o.unsqueeze(2))
    return torch.cat(outs, dim=2), None

def delta_rule_closed_form(q, k, v, g=None, scale=1.0):
    # q, k unused, v: [B,H,N,D], g: [B,H,N,D]
    if g is None:
        eta = torch.zeros_like(v)
    else:
        eta = torch.sigmoid(g)
    B, H, N, D = v.shape
    one_minus_eta = 1 - eta
    log_1m_eta = torch.log(one_minus_eta + 1e-8)
    log_cumsum = torch.cumsum(log_1m_eta, dim=2)
    log_cumsum_t = log_cumsum.unsqueeze(3)
    log_cumsum_j = log_cumsum.unsqueeze(2)
    prod = torch.exp(log_cumsum_t - log_cumsum_j)
    mask = torch.tril(torch.ones(N, N, device=v.device), diagonal=0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    prod = prod * mask
    ev = eta * v
    ev_j = ev.unsqueeze(2)
    h = torch.sum(prod * ev_j, dim=3)
    return h, None

class FLAOperator(nn.Module):
    """Unified FLA operator: GLA & delta_rule"""
    def __init__(self, mode="gla", head_dim=None):
        super().__init__()
        self.mode = mode.lower()
        self.head_dim = head_dim

    def forward(self, q, k, v, g=None, scale=1.0, initial_state=None, training=True, **kwargs):
        if self.mode == "gla":
            if TRITON_AVAILABLE:
                if training or q.shape[-2] > 1:
                    return fused_chunk_gla(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
                else:
                    return fused_recurrent_gla(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
            else:
                return gla_recurrent(q, k, v, g, scale)
        elif self.mode == "delta_rule":
            if TRITON_AVAILABLE:
                if training or q.shape[-2] > 1:
                    return fused_chunk_delta_rule(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
                else:
                    return fused_recurrent_delta_rule(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
            else:
                return delta_rule_closed_form(q, k, v, g, scale)
        else:
            raise ValueError(f"Unknown FLA operator: {self.mode}")

def get_fla_operator(name, head_dim=None):
    if isinstance(name, str):
        return FLAOperator(mode=name, head_dim=head_dim)
    elif callable(name):
        return name
    else:
        raise ValueError(f"Unknown FLA operator: {name}")