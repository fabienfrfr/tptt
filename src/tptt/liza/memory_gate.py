"""Linear Attention module for LiZA."""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..utils import Cache
from .mapping_func import get_attention_operator
from .utils import apply_attention_mask, repeat_kv, split_qkv


class LiZAttention(nn.Module):
    """LiZA Linear Attention module, mixing linear and vanilla attention."""

    def __init__(
        self,
        base_attn: nn.Module,
        layer_idx: int,
        config,  # PretrainedConfig
        cache: Cache = None,
        operator_mode: str = "delta_rule",
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
        use_rotary: bool = False,
    ):
        super().__init__()
        self.base_attn = base_attn
        self.config = config
        self.layer_idx = layer_idx
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size
        self.use_rotary = use_rotary
        self.cache = cache or Cache(max_length=getattr(config, "max_length", 2048))

        (
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            self.num_key_value_groups,
        ) = self.get_attention_parameters(base_attn, config)
        self.operator = get_attention_operator(operator_mode)
        self.pool_g = nn.AdaptiveAvgPool1d(
            output_size=self.head_dim * self.num_key_value_heads
        )

    def get_attention_parameters(self, base_attn, config):
        """Retrieve the attention parameters from the base attention module."""
        # first order base attention module and second order config
        num_heads = (
            getattr(base_attn, "num_heads", None)
            or getattr(base_attn, "num_q_heads", None)
            or getattr(config, "num_attention_heads", None)
        )
        head_dim = getattr(base_attn, "head_dim", None)
        num_key_value_heads = (
            getattr(base_attn, "num_kv_heads", None)
            or getattr(base_attn, "num_k_heads", None)
            or getattr(config, "num_key_value_heads", None)
            or num_heads  # fallback
        )
        num_key_value_groups = getattr(base_attn, "num_key_value_groups", None) or (
            num_heads // num_key_value_heads if num_heads and num_key_value_heads else 1
        )
        return num_heads, head_dim, num_key_value_heads, num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        print(f"hidden_states.shape: {hidden_states.shape}")
        base_attn = self.base_attn
        # 1. Dynamic retrieval of projections
        if hasattr(base_attn, "q_proj"):
            # LLama, OLMO and Mistral style
            q = base_attn.q_proj(hidden_states)
            k = base_attn.k_proj(hidden_states)
            v = base_attn.v_proj(hidden_states)
            out_proj = base_attn.o_proj
        elif hasattr(base_attn, "qkv_proj"):
            # OpenELM and GPT-Neo style : QKV fused, split on the last dimension
            qkv = base_attn.qkv_proj(hidden_states)
            q, k, v = split_qkv(base_attn, qkv)
            out_proj = base_attn.out_proj
        elif hasattr(base_attn, "c_attn") and hasattr(base_attn, "c_proj"):
            # GPT-2 style
            qkv = base_attn.c_attn(hidden_states)
            q, k, v = qkv.chunk(3, dim=-1)
            out_proj = base_attn.c_proj
        else:
            raise ValueError("Unsupported attention module: cannot find projections.")

        g = self.pool_g(k)

        # attention_mask: [batch, seq], v: [batch, seq, ...]
        if attention_mask is not None:
            v = apply_attention_mask(attention_mask, v)

        # 4. Reshape for multi-head
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_key_value_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_key_value_heads)
        g = rearrange(g, "b n (h m) -> b h n m", h=self.num_key_value_heads)

        # Repeat for GQA
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)

        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)

        gate_logit_normalizer = 16
        g = F.logsigmoid(g) / gate_logit_normalizer

        q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))
        batch_size, num_heads, seq_len, head_dim = q.shape

        if self.use_rotary:
            print(
                "[LinearAttention] Applying rotary embedding (not implemented in this snippet)"
            )

        # Retrieve recurrent state from cache
        last_state = self.cache[self.layer_idx]
        recurrent_state = (
            last_state["recurrent_state"]
            if last_state is not None and "recurrent_state" in last_state
            else None
        )

        # Linear attention
        o_lin, recurrent_state = self.operator(
            q,
            k,
            v,
            beta=g,
            chunk_size=self.max_chunk_size,
            recurrent_state=recurrent_state,
        )
        o_lin = o_lin.reshape(batch_size, num_heads, seq_len, head_dim)
        o_lin = rearrange(o_lin, "b h n d -> b n (h d)")
        o_lin = out_proj(o_lin)

        # Save recurrent state
        self.cache.update(self.layer_idx, recurrent_state=recurrent_state)

        # Standard attention
        o_base, attn_weights = self.base_attn(
            hidden_states, attention_mask=attention_mask, **kwargs
        )

        out = self.mag_weight * o_lin + (1 - self.mag_weight) * o_base
        return out, attn_weights
