from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import AttentionConfig
from .fla import get_fla_operator
from .utils import repeat_kv


# Memory Parallel Attention (MPA)
class ParallelFLAAttention(nn.Module):
    def __init__(
        self,
        base_attn: nn.Module,
        config: AttentionConfig,
        operator: Union[str, Callable] = "delta_rule",
        combine_fn: Optional[Callable] = None,
        operator_kwargs: Optional[Dict] = None,
        fla_weight: float = 0.01,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.base_attn = base_attn
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

        self.rotary_emb = getattr(base_attn, "rotary_emb", None)
        # self.pool_g = nn.AdaptiveAvgPool1d(output_size=self.head_dim * self.num_key_value_heads)
        self.g_proj = nn.Linear(self.head_dim, 1)
        self.operator = get_fla_operator(operator, head_dim=self.head_dim)
        self.combine_fn = combine_fn or (lambda orig, fla: orig + fla)
        self.operator_kwargs = operator_kwargs or {}
        self.fla_weight = fla_weight

    def _project_and_prepare(self, hidden_states, attention_mask):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(-1)
            v = v * attention_mask

        # Reshape q, k, v: [B, N, hidden_size] -> [B, N, H, D]
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_key_value_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_key_value_heads)

        # Gate projection
        g = self.g_proj(k)  # [B, N, H, D] -> [B, N, H, 1]
        g = g.expand(-1, -1, -1, self.head_dim)  # [B, N, H, D]

        # Rearrange to [B, H, N, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        g = g.permute(0, 2, 1, 3)

        # repeat_kv si besoin
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)
        return q, k, v, g

    def _preprocess_for_pla(self, q, k, v, g):
        """Preprocess the inputs for the parallel linear attention operator."""
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)
        g = F.logsigmoid(g) / 16
        q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))
        return q, k, v, g

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Dict] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        # --- Cache ---
        last_state = None
        if (
            past_key_value is not None
            and self.layer_idx is not None
            and len(past_key_value) > self.layer_idx
        ):
            last_state = past_key_value[self.layer_idx]

        # --- Projections & reshape ---
        q, k, v, g = self._project_and_prepare(hidden_states, attention_mask)
        q, k, v, g = self._preprocess_for_pla(q, k, v, g)
        scale = 1.0 / (self.head_dim**0.5)

        # --- FLA operator ---
        fla_out, new_recurrent_state = self.operator(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=last_state,
            training=self.training,
            **self.operator_kwargs
        )

        if past_key_value is not None and self.layer_idx is not None:
            past_key_value.update(
                recurrent_state=new_recurrent_state,
                layer_idx=self.layer_idx,
                offset=q.shape[1],
            )

        # --- Rotary embeddings (optionnel) ---
        if self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(v, position_ids)
            # Optionally: apply rotary to q/k

        # --- Combine ---
        fla_out = rearrange(fla_out, "b h n d -> b n (h d)")
        fla_out = fla_out.to(hidden_states.dtype)
        base_out, attn_weights = self.base_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        combined = self.fla_weight * fla_out + (1 - self.fla_weight) * base_out
        return combined, attn_weights
