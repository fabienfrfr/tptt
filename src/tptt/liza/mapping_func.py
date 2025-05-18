"""Attention operator module for linear and GLA attention in PyTorch."""

import torch
from torch import nn

if torch.cuda.is_available():
    from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
else:
    fused_chunk_gla = None  # pylint: disable=invalid-name
    fused_recurrent_gla = None  # pylint: disable=invalid-name

from .utils import get_valid_chunk_size


class AttentionOperator(nn.Module):
    """Base class for linear attention operators."""

    def __init__(self, mode="delta_rule", head_dim=None, training=False):
        super().__init__()
        self.mode = mode
        self.head_dim = head_dim
        self.training = training

    def forward(self, q, k, v, **options):
        """Forward pass for the attention operator."""
        beta = options.get("beta", None)
        chunk_size = options.get("chunk_size", 64)
        scale = options.get("scale", 1)
        recurrent_state = options.get("recurrent_state", None)

        if self.mode == "delta_rule":
            return self.chunk_delta_rule_forward(
                q, k, v, beta, chunk_size, initial_state=recurrent_state
            )
        if self.mode == "gla":
            if fused_chunk_gla is None or fused_recurrent_gla is None:
                raise RuntimeError("GLA kernels are not available: CUDA required.")
            if self.training or q.shape[-2] > 1:
                return fused_chunk_gla(
                    q,
                    k,
                    v,
                    beta,
                    scale=scale,
                    initial_state=recurrent_state,
                    output_final_state=True,
                )
            return fused_recurrent_gla(
                q,
                k,
                v,
                beta,
                scale=scale,
                initial_state=recurrent_state,
                output_final_state=True,
            )
        raise ValueError(f"Unknown operator mode: {self.mode}")

    @staticmethod
    def chunk_delta_rule_forward(
        query, key, value, beta, chunk_size, initial_state=None
    ):
        """Chunkwise delta rule attention computation."""

        batch_size, num_heads, seq_len, head_dim = query.shape
        total_length = batch_size * num_heads * seq_len

        # Flatten for operator: [batch_size * num_heads * seq_len, head_dim]
        q_lin = query.reshape(total_length, head_dim)
        k_lin = key.reshape(total_length, head_dim)
        v_lin = value.reshape(total_length, head_dim)
        g_lin = beta.reshape(total_length, head_dim)

        # Update with Validate chunk size
        chunk_size = get_valid_chunk_size(total_length, chunk_size)
        num_chunks = total_length // chunk_size

        # Reshaping for chunk
        query = q_lin.reshape(num_chunks, chunk_size, head_dim)
        key = k_lin.reshape(num_chunks, chunk_size, head_dim)
        value = v_lin.reshape(num_chunks, chunk_size, head_dim)
        beta = g_lin.reshape(num_chunks, chunk_size, head_dim)

        key_beta = key * beta
        value_beta = value * beta

        output = torch.empty_like(value)
        # initialize the state
        state = (
            initial_state
            if initial_state is not None
            else torch.zeros(head_dim, head_dim, device=query.device, dtype=query.dtype)
        )

        def process_chunk(query_i, key_i, key_beta_i, value_beta_i, state):
            """Process a single chunk with closed form delta rule"""
            t_matrix = -(key_beta_i @ key_i.t()).tril(-1)
            t_matrix = t_matrix + torch.eye(
                chunk_size, device=query.device, dtype=query.dtype
            )
            w_matrix = t_matrix @ key_beta_i
            u_matrix = t_matrix @ value_beta_i
            u_i = u_matrix - w_matrix @ state
            o_inter = query_i @ state
            a_i = (query_i @ key_i.t()).tril()
            o_intra = a_i @ u_i
            state = state + key_i.t() @ u_i
            return o_intra + o_inter, state

        for i in range(num_chunks):
            # Process a all chunk with recurrent form delta rule
            chunk_out, state = process_chunk(
                query[i], key[i], key_beta[i], value_beta[i], state
            )
            output[i] = chunk_out
        return output.reshape(total_length, head_dim), state.reshape(head_dim, head_dim)


def get_attention_operator(mode, head_dim=None):
    """Factory for AttentionOperator."""
    return AttentionOperator(mode=mode, head_dim=head_dim)
