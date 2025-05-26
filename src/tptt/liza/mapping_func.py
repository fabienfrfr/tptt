"""Attention operator module for linear and GLA attention in PyTorch."""

import torch
from torch import nn

fused_chunk_gla = None  # pylint: disable=invalid-name
fused_recurrent_gla = None  # pylint: disable=invalid-name

if torch.cuda.is_available():
    try:
        from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
    except ImportError:
        pass

from .utils import get_valid_chunk_size


class AttentionOperator(nn.Module):
    """Base class for linear attention operators."""

    def __init__(self, mode="delta_rule"):
        super().__init__()
        self.mode = mode

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
            return self.gla_forward(q, k, v, beta, scale, initial_state=recurrent_state)
        raise ValueError(f"Unknown operator mode: {self.mode}")

    @staticmethod
    def chunk_delta_rule_forward(
        query, key, value, beta, chunk_size, initial_state=None
    ):
        """
        Implementation of https://arxiv.org/abs/2406.06484
        query, key, value, beta: [batch, num_heads, seq_len, head_dim]
        chunk_size: int
        initial_state: [batch, num_heads, head_dim, head_dim] or None
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        print("query shape : ", query.shape)
        chunk_size = get_valid_chunk_size(seq_len, chunk_size)
        num_chunks = seq_len // chunk_size

        # Reshape for chunking: [batch, num_heads, num_chunks, chunk_size, head_dim]
        q_chunks = query.reshape(
            batch_size, num_heads, num_chunks, chunk_size, head_dim
        )
        k_chunks = key.reshape(batch_size, num_heads, num_chunks, chunk_size, head_dim)
        v_chunks = value.reshape(
            batch_size, num_heads, num_chunks, chunk_size, head_dim
        )
        beta_chunks = beta.reshape(
            batch_size, num_heads, num_chunks, chunk_size, head_dim
        )

        # Output buffer
        output = torch.empty_like(q_chunks)
        # State: [batch, num_heads, head_dim, head_dim]
        if initial_state is not None:
            state = initial_state
        else:
            state = torch.zeros(
                batch_size,
                num_heads,
                head_dim,
                head_dim,
                device=query.device,
                dtype=query.dtype,
            )

        def process_chunk(q, k, v, b, state):
            """
            q, k, v, b: [batch, num_heads, chunk_size, head_dim]
            state: [batch, num_heads, head_dim, head_dim]
            Returns: (output_chunk, new_state)
            """
            # Clamp to avoid numerical instabilities (not in paper)
            k = torch.clamp(k, min=-1e4, max=1e4)
            v = torch.clamp(v, min=-1e4, max=1e4)
            b = torch.clamp(b, min=1e-6, max=1e4)
            q = torch.clamp(q, min=-1e4, max=1e4)

            # Eq. (10): β_t * k_t and β_t * v_t
            k_beta = k * b
            v_beta = v * b

            # Eq. (11): Lower-triangular matrix T (with -KβK^T off-diagonal, 1 on diagonal)
            # T = I - tril(KβK^T, -1)
            t_matrix = -(k_beta @ k.transpose(-2, -1)).tril(-1)
            t_matrix = t_matrix + torch.eye(
                q.shape[-2], device=q.device, dtype=q.dtype
            ).unsqueeze(0).unsqueeze(0)

            # Eq. (11): W = T Kβ, U = T Vβ
            w_matrix = t_matrix @ k_beta
            u_matrix = t_matrix @ v_beta

            # Eq. (12): u_i = U - W S (S = state)
            print("w_matrix : ", w_matrix.shape)
            print("state : ", state.shape)
            u_i = u_matrix - torch.matmul(w_matrix, state)

            # Eq. (12): inter-chunk output: q S
            o_inter = torch.matmul(q, state)

            # Eq. (12): intra-chunk attention: tril(q K^T)
            a_i = (q @ k.transpose(-2, -1)).tril()

            # Eq. (12): intra-chunk output: a_i u_i
            o_intra = torch.matmul(a_i, u_i)

            # Eq. (12): state update: S_new = S + K^T u_i
            new_state = state + torch.matmul(k.transpose(-2, -1), u_i)

            # Eq. (12): output = intra + inter
            return o_intra + o_inter, new_state

        for chunk_idx in range(num_chunks):
            q = q_chunks[:, :, chunk_idx]
            k = k_chunks[:, :, chunk_idx]
            v = v_chunks[:, :, chunk_idx]
            b = beta_chunks[:, :, chunk_idx]

            chunk_out, state = process_chunk(q, k, v, b, state)
            output[:, :, chunk_idx] = chunk_out

        # Reshape back to [batch, num_heads, seq_len, head_dim]
        output = output.reshape(batch_size, num_heads, seq_len, head_dim)
        print("output shape : ", output.shape)
        print("state shape : ", state.shape)
        return output, state

    @staticmethod
    def gla_forward(q, k, v, beta, scale, initial_state=None):
        """Forward pass for GLA attention operator."""
        if fused_chunk_gla is None or fused_recurrent_gla is None:
            raise RuntimeError("GLA kernels are not available: CUDA required.")
        if q.shape[-2] > 1:
            # Training or sequence length > 1
            return fused_chunk_gla(
                q,
                k,
                v,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
            )
        return fused_recurrent_gla(
            q,
            k,
            v,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )


def get_attention_operator(mode):
    """Factory for AttentionOperator."""
    return AttentionOperator(mode=mode)
