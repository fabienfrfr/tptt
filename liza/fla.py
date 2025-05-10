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
        warnings.warn(f"Triton is not available: {e}. Falling back to CPU.")
        return False

TRITON_AVAILABLE = check_triton_availability()

if TRITON_AVAILABLE:
    from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
    from fla.ops.delta_rule import fused_chunk_delta_rule, fused_recurrent_delta_rule

def gla_scan(q, k, v, g=None, scale=1.0):
    # q, k, v: [B, H, N, D]
    # g: [B, H, N, D] or None
    B, H, N, D = q.shape
    q = q * scale
    if g is None:
        g = torch.zeros_like(q)

    def step(carry, inputs):
        s_tm1 = carry  # [B, H, D]
        q_t, k_t, v_t, g_t = inputs
        s_t = s_tm1 + torch.sigmoid(q_t * k_t + g_t) * v_t
        y_t = s_t
        return s_t, y_t

    s0 = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    q_seq = q.transpose(2, 0)
    k_seq = k.transpose(2, 0)
    v_seq = v.transpose(2, 0)
    g_seq = g.transpose(2, 0)
    inputs = (q_seq, k_seq, v_seq, g_seq)
    _, y = torch.scan(
        fn=step,
        inputs=inputs,
        initial=s0,
        length=q_seq.shape[0]
    )
    y = y.transpose(0, 2).transpose(0, 1)
    return y, None

def delta_rule_scan(q, k, v, g=None, scale=1.0):
    B, H, N, D = q.shape
    q = q * scale
    if g is None:
        g = torch.zeros_like(q)

    def step(carry, inputs):
        s_tm1 = carry
        q_t, k_t, v_t, g_t = inputs
        delta = torch.sigmoid(q_t * k_t + g_t) * (v_t - s_tm1)
        s_t = s_tm1 + delta
        y_t = s_t
        return s_t, y_t

    s0 = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    q_seq = q.transpose(2, 0)
    k_seq = k.transpose(2, 0)
    v_seq = v.transpose(2, 0)
    g_seq = g.transpose(2, 0)
    inputs = (q_seq, k_seq, v_seq, g_seq)
    _, y = torch.scan(
        fn=step,
        inputs=inputs,
        initial=s0,
        length=q_seq.shape[0]
    )
    y = y.transpose(0, 2).transpose(0, 1)
    return y, None

def attention_rnn_scan(q, k, v, scale=1.0):
    # q, k, v: [B, H, N, D]
    B, H, N, D = q.shape
    q = q * scale

    def step(carry, inputs):
        n_tm1, d_tm1, m_tm1 = carry
        q_t, k_t, v_t = inputs
        alpha_t = torch.sum(q_t * k_t, dim=-1, keepdim=True)
        m_t = torch.maximum(m_tm1, alpha_t)
        exp_mdiff = torch.exp(m_tm1 - m_t)
        exp_adiff = torch.exp(alpha_t - m_t)
        d_t = exp_mdiff * d_tm1 + exp_adiff
        n_t = exp_mdiff * n_tm1 + exp_adiff * v_t
        y_t = n_t / d_t
        return (n_t, d_t, m_t), y_t

    n0 = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    d0 = torch.zeros(B, H, 1, dtype=q.dtype, device=q.device)
    m0 = torch.full((B, H, 1), float('-inf'), dtype=q.dtype, device=q.device)

    q_seq = q.transpose(2, 0)
    k_seq = k.transpose(2, 0)
    v_seq = v.transpose(2, 0)
    inputs = (q_seq, k_seq, v_seq)

    _, y = torch.scan(
        fn=step,
        inputs=inputs,
        initial=(n0, d0, m0),
        length=q_seq.shape[0]
    )
    y = y.transpose(0, 2).transpose(0, 1)
    return y, None

class FLAOperator(nn.Module):
    """Unified FLA operator: GLA, delta_rule, attention_rnn (Aaren), etc."""
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
                return gla_scan(q, k, v, g, scale)
        elif self.mode == "delta_rule":
            if TRITON_AVAILABLE:
                if training or q.shape[-2] > 1:
                    return fused_chunk_delta_rule(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
                else:
                    return fused_recurrent_delta_rule(q, k, v, g, scale=scale, initial_state=initial_state, output_final_state=True, **kwargs)
            else:
                return delta_rule_scan(q, k, v, g, scale)
        elif self.mode in ["attention_rnn", "aaren"]:
            return attention_rnn_scan(q, k, v, scale)
        else:
            raise ValueError(f"Unknown FLA operator: {self.mode}")

def get_fla_operator(name, head_dim=None):
    if isinstance(name, str):
        return FLAOperator(mode=name, head_dim=head_dim)
    elif callable(name):
        return name
    else:
        raise ValueError(f"Unknown FLA operator: {name}")
