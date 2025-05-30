"""Linear Attention module for LiZA."""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..utils import LCache
from .mapping_func import get_attention_operator
from .utils import (apply_linear_attention_mask, repeat_kv, split_qkv,
                    truncate_attention_mask)

# from transformers.cache_utils import DynamicCache




class LiZAttention(nn.Module):
    """LiZA Linear Attention module, mixing linear and vanilla attention."""

    def __init__(
        self,
        base_attn: nn.Module,
        layer_idx: int,
        config,  # PretrainedConfig
        linear_cache: LCache = None,
        operator_mode: str = "delta_rule",
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
    ):
        super().__init__()
        self.base_attn = base_attn
        self.config = config
        self.layer_idx = layer_idx
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size
        self.attn_ratio = getattr(config, "attn_ratio", 0.5)
        self.linear_cache = linear_cache or LCache()
        (
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.max_attn_length,
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
        max_attn_length = (
            getattr(config, "max_length", None)
            or getattr(config, "model_max_length", None)
            or getattr(base_attn, "max_position_embeddings", None)
            or 2048  # default fallback
        )
        return (
            num_heads,
            head_dim,
            num_key_value_heads,
            num_key_value_groups,
            max_attn_length,
        )

    def apply_projections(self, hidden_states):
        base_attn = self.base_attn
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
        return q, k, v, out_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        device = hidden_states.device
        self.base_attn.to(device)
        if self.training:
            kwargs.pop("past_key_value", None)
            kwargs["use_cache"] = False
        else:
            # Generation/Inference mode (incremental decoding)
            if kwargs.get("past_key_value", None) is not None:
                kwargs["use_cache"] = True
            # Evaluation/Validation mode
            else:
                kwargs["use_cache"] = False

        # Apply projections to hidden states
        q, k, v, out_proj = self.apply_projections(hidden_states)
        g = self.pool_g(k)

        # Manage attention mask (with padding)
        if attention_mask is not None:
            # attention_mask -> [batch, seq], v: [batch, seq, ...]
            v = apply_linear_attention_mask(attention_mask, v)

        # Reshape for multi-head
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_key_value_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_key_value_heads)
        g = rearrange(g, "b n (h m) -> b h n m", h=self.num_key_value_heads)

        # Repeat for GQA
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)

        q = torch.clamp(F.softmax(q, dim=-1), min=1e-6, max=1 - 1e-6)
        k = torch.clamp(F.softmax(k, dim=-1), min=1e-6, max=1 - 1e-6)

        gate_norm = kwargs.get("gate_logit_normalizer", 16)
        g = F.logsigmoid(g) / gate_norm
        g = torch.clamp(g, min=-gate_norm, max=gate_norm)

        # Convert to float32 for numerical stability and get model dtype
        model_dtype = q.dtype
        q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))

        # Retrieve recurrent state from cache (inference only)
        if kwargs["use_cache"]:
            last_state = self.linear_cache[self.layer_idx]
            recurrent_state = (
                last_state["recurrent_state"]
                if last_state is not None and "recurrent_state" in last_state
                else None
            )
        else:
            recurrent_state = None

        # Linear attention
        o_lin, recurrent_state = self.operator(
            q,
            k,
            v,
            beta=g,
            chunk_size=self.max_chunk_size,
            recurrent_state=recurrent_state,
        )
        o_lin = rearrange(o_lin, "b h n d -> b n (h d)").to(model_dtype)
        o_lin = out_proj(o_lin).to(model_dtype)  # force to model dtype

        # Save recurrent state
        if kwargs["use_cache"]:
            self.linear_cache.update(self.layer_idx, recurrent_state=recurrent_state)

        # If cache_implementation="static" -> truncated attention
        if kwargs["use_cache"]:
            hidden_states, attention_mask = truncate_attention_mask(
                hidden_states, attention_mask, self.max_attn_length
            )
        # Standard attention (mask and rotation is applied inside)
        base_attn_outputs = self.base_attn(
            hidden_states, attention_mask=attention_mask, **kwargs
        )
        if isinstance(base_attn_outputs, tuple):
            if len(base_attn_outputs) == 3:
                o_base, attn_weights, present_key_value = base_attn_outputs
            elif len(base_attn_outputs) == 2:
                o_base, attn_weights = base_attn_outputs
            else:
                raise ValueError(
                    f"Unexpected number of outputs from base_attn: {len(base_attn_outputs)}"
                )
        else:
            o_base = base_attn_outputs

        # Apply Memory as Gate in self-attention (with model_max_length management)
        if o_lin.shape[1] != o_base.shape[1]:
            left_trunc = min(self.max_attn_length, o_lin.shape[1], o_base.shape[1])
            o_lin, o_base = o_lin[:, -left_trunc:], o_base[:, -left_trunc:]
        out = self.mag_weight * o_lin + (1 - self.mag_weight) * o_base.to(model_dtype)

        # Return output following transformer convention
        if isinstance(base_attn_outputs, tuple):
            if len(base_attn_outputs) == 3:
                return out, attn_weights, present_key_value
            elif len(base_attn_outputs) == 2:
                return out, attn_weights
        else:
            return out
