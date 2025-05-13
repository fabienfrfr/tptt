from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention_operator import get_attention_operator


class LinearAttention(nn.Module):
    def __init__(
        self,
        base_attn: nn.Module,
        config,  # PretrainedConfig
        operator_mode: str = "delta_rule",
        fla_weight: float = 0.5,
        chunk_size: int = 64,
        use_rotary: bool = False,
    ):
        super().__init__()
        self.base_attn = base_attn

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.embed_dim = config.hidden_size
        self.fla_weight = fla_weight
        self.chunk_size = chunk_size

        # Projections partagées
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

        self.use_rotary = use_rotary

        self.pool_g = nn.AdaptiveAvgPool1d(
            output_size=self.head_dim * self.num_key_value_heads
        )
        self.operator = get_attention_operator(operator_mode, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.pool_g(k)

        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2] :, None])

        # Reshape pour multi-head
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_key_value_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_key_value_heads)
        g = rearrange(g, "b n (h m) -> b h n m", h=self.num_key_value_heads)

        # Répétition pour GQA
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)

        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)

        gate_logit_normalizer = 16
        g = F.logsigmoid(g) / gate_logit_normalizer

        q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))

        if self.use_rotary:
            print(
                "[LinearAttention] Applying rotary embedding (not implemented in this snippet)"
            )

        B, H, N, D_h = q.shape

        # On a besoin de [L, d] pour l'opérateur, L = B * H * N, d = D_h
        q_lin = q.reshape(B * H * N, D_h)
        k_lin = k.reshape(B * H * N, D_h)
        v_lin = v.reshape(B * H * N, D_h)
        g_lin = g.reshape(B * H * N, D_h)

        # Vérification chunk_size
        total_L = B * H * N
        valid_chunk_size = get_valid_chunk_size(total_L, self.chunk_size)

        # Attention linéaire
        o_lin, _ = self.operator(
            q_lin, k_lin, v_lin, beta=g_lin, chunk_size=valid_chunk_size
        )
        o_lin = o_lin.reshape(B, H, N, D_h)
        o_lin = rearrange(o_lin, "b h n d -> b n (h d)")
        o_lin = self.o_proj(o_lin)

        # Attention standard
        o_base, attn_weights = self.base_attn(
            hidden_states, attention_mask=attention_mask, **kwargs
        )

        out = self.fla_weight * o_lin + (1 - self.fla_weight) * o_base
        return out, attn_weights
