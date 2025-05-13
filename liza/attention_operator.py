import torch
import torch.nn as nn


class AttentionOperator(nn.Module):
    """Base class for linear attention operators."""

    def __init__(self, mode="delta_rule", head_dim=None):
        super().__init__()
        self.mode = mode
        self.head_dim = head_dim

    def forward(self, q, k, v, beta=None, chunk_size=64, initial_state=None, **kwargs):
        if self.mode == "delta_rule":
            return self.chunk_delta_rule_forward(q, k, v, beta, chunk_size)
        elif self.mode == "gla":
            raise NotImplementedError("GLA not implemented yet.")
        else:
            raise ValueError(f"Unknown operator mode: {self.mode}")

    @staticmethod
    def chunk_delta_rule_forward(Q, K, V, beta, C):
        L, d = Q.shape
        n_chunks = L // C

        Q = Q.reshape(n_chunks, C, d)
        K = K.reshape(n_chunks, C, d)
        V = V.reshape(n_chunks, C, d)
        beta = beta.reshape(n_chunks, C, d)

        K_beta = K * beta  # [n_chunks, C, d]
        V_beta = V * beta  # [n_chunks, C, d]

        O = torch.empty_like(V)
        S = torch.zeros(d, d, device=Q.device, dtype=Q.dtype)
        for i in range(n_chunks):
            q_i = Q[i]  # [C, d]
            k_i = K[i]  # [C, d]
            v_i = V[i]  # [C, d]
            k_beta_i = K_beta[i]  # [C, d]
            v_beta_i = V_beta[i]  # [C, d]

            # Eq. 10: Compute T (lower-triangular)
            T = -(k_beta_i @ k_i.t()).tril(-1)  # [C, C]
            T = T + torch.eye(C, device=Q.device, dtype=Q.dtype)
            # Eq. 11: W, U
            W = T @ k_beta_i  # [C, d]
            U = T @ v_beta_i  # [C, d]
            # Eq. 8-9: chunkwise parallel
            u_i = U - W @ S  # [C, d]
            o_inter = q_i @ S  # [C, d]
            A_i = (q_i @ k_i.t()).tril()  # [C, C]
            o_intra = A_i @ u_i  # [C, d]
            S = S + k_i.t() @ u_i  # [d, d]
            O[i] = o_intra + o_inter
        return O.reshape(L, d), None


def get_attention_operator(mode, head_dim=None):
    return AttentionOperator(mode=mode, head_dim=head_dim)
