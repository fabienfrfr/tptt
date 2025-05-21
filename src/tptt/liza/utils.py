"""Utility functions for LiZA attention."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped query attention (GQA)."""
    return x.repeat_interleave(n_rep, dim=1)


def split_qkv(base_attn, qkv):
    """Split the QKV tensor into separate Q, K, and V tensors."""
    num_q_heads = getattr(base_attn, "num_q_heads", None)
    num_k_heads = getattr(base_attn, "num_k_heads", None)
    num_v_heads = getattr(base_attn, "num_v_heads", None)
    head_dim = getattr(base_attn, "head_dim", None)

    q_len = num_q_heads * head_dim
    k_len = num_k_heads * head_dim
    v_len = num_v_heads * head_dim

    q, k, v = torch.split(qkv, [q_len, k_len, v_len], dim=-1)
    return q, k, v


def apply_attention_mask(attention_mask, v):
    logging.info(f"attention_mask.shape: {attention_mask.shape}")
    logging.info(f"v.shape: {v.shape}")
    # Squeeze all singleton dims except batch (dim=0)
    mask = attention_mask.squeeze(
        dim=tuple(
            i for i in range(1, attention_mask.dim()) if attention_mask.shape[i] == 1
        )
    )
    # handle left padding : mask is [batch, seq] --> Broadcast to v
    mask = mask[:, -v.shape[-2] :][(...,) + (None,) * (v.dim() - 2)]
    return v * mask


def get_valid_chunk_size(total_l: int, chunk_size: int) -> int:
    """
    Return the largest chunk_size <= chunk_size that divides total_l.
    If no chunk_size > 1 fits, return 1.
    """
    for c in range(min(chunk_size, total_l), 0, -1):
        if total_l % c == 0:
            return c
    return 1


def match_dim(x: torch.Tensor, dim: int, target_size: int) -> torch.Tensor:
    """
    Match the size of tensor x along dimension dim to target_size by interpolation
    or projection.
    """
    src_size = x.shape[dim]
    if src_size == target_size:
        return x
    x = torch.moveaxis(x, dim, -1)
    shape = x.shape
    if src_size < target_size:
        x = x.reshape(-1, 1, src_size)
        x = F.interpolate(x, size=target_size, mode="linear", align_corners=False)
        x = x.reshape(*shape[:-1], target_size)
    else:
        eye = torch.eye(target_size, src_size, device=x.device, dtype=x.dtype)
        x = F.linear(x, eye)  # pylint: disable=not-callable
    x = torch.moveaxis(x, -1, dim)
    return x
