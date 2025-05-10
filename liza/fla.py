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
        warnings.warn(f"Triton is not available: {e}. Falling back to JAX.")
        return False

TRITON_AVAILABLE = check_triton_availability()

if TRITON_AVAILABLE:
    from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
    from fla.ops.delta_rule import fused_chunk_delta_rule, fused_recurrent_delta_rule
else:
    import jax
    import jax.numpy as jnp

    def gla_scan(q, k, v, g=None, scale=1.0):
        # q, k, v: [B, H, N, D] (torch.Tensor, convert to jax)
        q, k, v = map(lambda x: jax.device_put(jnp.array(x.cpu().numpy())), (q, k, v))
        B, H, N, D = q.shape
        q = q * scale
        if g is None:
            g = jnp.zeros((B, H, N, 1), dtype=q.dtype)
        else:
            g = jax.device_put(jnp.array(g.cpu().numpy()))
        lambda_t = jax.nn.sigmoid(g)

        def step(S_tm1, inputs):
            q_t, k_t, v_t, lambda_t_ = inputs
            kvT = jnp.einsum('bhd,bhe->bhde', k_t, v_t)
            S_t = lambda_t_ * S_tm1 + (1 - lambda_t_) * kvT
            o_t = jnp.einsum('bhd,bhde->bhe', q_t, S_t)
            return S_t, o_t

        S0 = jnp.zeros((B, H, D, D), dtype=q.dtype)
        q_seq = jnp.transpose(q, (2, 0, 1, 3))
        k_seq = jnp.transpose(k, (2, 0, 1, 3))
        v_seq = jnp.transpose(v, (2, 0, 1, 3))
        lambda_seq = jnp.transpose(lambda_t, (2, 0, 1, 3))
        (_, o) = jax.lax.scan(step, S0, (q_seq, k_seq, v_seq, lambda_seq))
        o = jnp.transpose(o, (1, 2, 0, 3))
        return torch.from_numpy(np.array(o)), None

    def delta_rule_scan(q, k, v, g=None, scale=1.0):
        q, v = map(lambda x: jax.device_put(jnp.array(x.cpu().numpy())), (q, v))
        B, H, N, D = q.shape
        if g is None:
            g = jnp.zeros((B, H, N, D), dtype=q.dtype)
        else:
            g = jax.device_put(jnp.array(g.cpu().numpy()))
        eta_t = jax.nn.sigmoid(g)

        def step(h_tm1, inputs):
            v_t, eta_t_ = inputs
            h_t = h_tm1 + eta_t_ * (v_t - h_tm1)
            return h_t, h_t

        h0 = jnp.zeros((B, H, D), dtype=q.dtype)
        v_seq = jnp.transpose(v, (2, 0, 1, 3))
        eta_seq = jnp.transpose(eta_t, (2, 0, 1, 3))
        _, h = jax.lax.scan(step, h0, (v_seq, eta_seq))
        h = jnp.transpose(h, (1, 2, 0, 3))
        return torch.from_numpy(np.array(h)), None

    def attention_rnn_scan(q, k, v, scale=1.0):
        q, k, v = map(lambda x: jax.device_put(jnp.array(x.cpu().numpy())), (q, k, v))
        B, H, N, D = q.shape
        q = q * scale

        def step(carry, inputs):
            n_tm1, d_tm1, m_tm1 = carry
            q_t, k_t, v_t = inputs
            alpha_t = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
            m_t = jnp.maximum(m_tm1, alpha_t)
            exp_mdiff = jnp.exp(m_tm1 - m_t)
            exp_adiff = jnp.exp(alpha_t - m_t)
            d_t = exp_mdiff * d_tm1 + exp_adiff
            n_t = exp_mdiff * n_tm1 + exp_adiff * v_t
            y_t = n_t / d_t
            return (n_t, d_t, m_t), y_t

        n0 = jnp.zeros((B, H, D), dtype=q.dtype)
        d0 = jnp.zeros((B, H, 1), dtype=q.dtype)
        m0 = jnp.full((B, H, 1), -jnp.inf, dtype=q.dtype)

        q_seq = jnp.transpose(q, (2, 0, 1, 3))
        k_seq = jnp.transpose(k, (2, 0, 1, 3))
        v_seq = jnp.transpose(v, (2, 0, 1, 3))
        (_, _, _), y = jax.lax.scan(step, (n0, d0, m0), (q_seq, k_seq, v_seq))
        y = jnp.transpose(y, (1, 2, 0, 3))
        return torch.from_numpy(np.array(y)), None

import numpy as np

class FLAOperator(nn.Module):
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
