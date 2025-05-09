from typing import Optional, Callable, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .config import AttentionConfig
from .utils import repeat_kv
from .fla import gla_attention, delta_rule_attention

class ParallelFLAAttention(nn.Module):
    def __init__(
        self,
        base_attn: nn.Module,
        config: AttentionConfig,
        operator: Union[str, Callable] = "gla",
        combine_fn: Optional[Callable] = None,
        operator_kwargs: Optional[Dict] = None,
        fla_weight: float = 0.01,
    ):
        super().__init__()
        self.base_attn = base_attn
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

        self.operator = operator
        self.combine_fn = combine_fn or (lambda orig, fla: orig + fla)
        self.operator_kwargs = operator_kwargs or {}
        self.fla_weight = fla_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_key_value_heads)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / (self.head_dim ** 0.5)

        # --- Attention alternative : GLA ou delta_rule ---
        if isinstance(self.operator, str):
            if self.operator == "gla":
                fla_out = gla_attention(q, k, v, scale=scale, **self.operator_kwargs)
            elif self.operator == "delta_rule":
                fla_out = delta_rule_attention(q, k, v, scale=scale, **self.operator_kwargs)
            else:
                raise ValueError(f"Unknown FLA operator: {self.operator}")
        elif callable(self.operator):
            fla_out = self.operator(q, k, v, scale=scale, **self.operator_kwargs)
        else:
            raise ValueError("Unknown operator for FLA")

        fla_out = rearrange(fla_out, 'b h n d -> b n (h d)')
        fla_out = fla_out.to(hidden_states.dtype)

        # --- Attention standard ---
        base_out, attn_weights = self.base_attn(hidden_states, attention_mask=attention_mask, **kwargs)

        # --- Combinaison ---
        combined = self.combine_fn(base_out, self.fla_weight * fla_out)
        return combined, attn_weights
