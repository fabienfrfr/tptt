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

def gla_recurrent(q, k, v, g=None, scale=1.0, initial_state=None):
    """
    S_t = lambda_t * S_{t-1} + (1 - lambda_t) * k_t * v_t^T
    """
    # q, k, v: [B, H, N, D], g: [B, H, N, 1], initial_state: [B, H, D, D] or None
    B, H, N, D = q.shape
    if g is None:
        lambd = torch.zeros(B, H, N, 1, dtype=q.dtype, device=q.device)
    else:
        lambd = torch.sigmoid(g)
    q = q * scale
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(B, H, D, D, dtype=q.dtype, device=q.device)
    outs = []
    for t in range(N):
        kvT = torch.einsum('bhd,bhe->bhde', k[:, :, t, :], v[:, :, t, :])
        lmbd = lambd[:, :, t, :].reshape(B, H, 1, 1)
        S = lmbd * S + (1 - lmbd) * kvT
        o = torch.einsum('bhd,bhde->bhe', q[:, :, t, :], S)
        outs.append(o.unsqueeze(2))
    return torch.cat(outs, dim=2), None


def delta_rule_closed_form(q, k, v, g=None, scale=1.0, initial_state=None):
    """ 
    recurrent : h_t = (1 - eta_t) * h_{t-1} + eta_t * v_t
    closed form : h_t = (prod_{i=1}^t (1 - eta_i)) * h_0 + sum_{j=1}^t [ eta_j * (prod_{i=j+1}^t (1 - eta_i)) * v_j ]
    closed form (h_0 = 0) : h_t = sum_{j=1}^t [ eta_j * (prod_{i=j+1}^t (1 - eta_i)) * v_j ]
    """
    # q, k unused, v: [B, H, N, D], g: [B, H, N, D], h0: [B, H, D] or None
    if g is None:
        eta = torch.zeros_like(v)
    else:
        eta = torch.sigmoid(g)
    B, H, N, D = v.shape
    one_minus_eta = 1 - eta
    log_1m_eta = torch.log(one_minus_eta + 1e-8)
    log_cumsum = torch.cumsum(log_1m_eta, dim=2)  # [B,H,N,D]
    log_cumsum_t = log_cumsum.unsqueeze(3)        # [B,H,N,1,D]
    log_cumsum_j = log_cumsum.unsqueeze(2)        # [B,H,1,N,D]
    prod = torch.exp(log_cumsum_t - log_cumsum_j) # [B,H,N,N,D]
    mask = torch.tril(torch.ones(N, N, device=v.device), diagonal=0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    prod = prod * mask
    ev = eta * v
    ev_j = ev.unsqueeze(2)                        # [B,H,1,N,D]
    h = torch.sum(prod * ev_j, dim=3)             # [B,H,N,D]
    if initial_state is not None:
        h0 = initial_state
        prod_h0 = torch.exp(log_cumsum)           # [B,H,N,D]
        h = h + prod_h0 * h0.unsqueeze(2)         # [B,H,N,D]
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