"""Linear Attention module for LiZA."""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .attention_operator import get_attention_operator
from .utils import get_valid_chunk_size, repeat_kv


class LiZAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """LiZA Linear Attention module, mixing linear and vanilla attention."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        base_attn: nn.Module,
        config,  # PretrainedConfig
        operator_mode: str = "delta_rule",
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
        use_rotary: bool = False,
    ):
        """
        Args:
            base_attn (nn.Module): Standard attention module.
            config: Model configuration.
            operator_mode (str): Mode for attention operator.
            mag_weight (float): Weight for mixing linear and vanilla attention.
            chunk_size (int): Chunk size for linear attention.
            use_rotary (bool): Whether to use rotary embeddings.
        """
        super().__init__()
        self.base_attn = base_attn

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.embed_dim = config.hidden_size
        self.mag_weight = mag_weight  # Memory as Gate Weight
        self.max_chunk_size = max_chunk_size

        # Shared projections
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

        self.use_rotary = use_rotary

        self.pool_g = nn.AdaptiveAvgPool1d(
            output_size=self.head_dim * self.num_key_value_heads
        )
        self.operator = get_attention_operator(operator_mode, self.head_dim)

    def forward(  # pylint: disable=too-many-locals
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for LiZA Linear Attention.
        Args:
            hidden_states (torch.Tensor): Input tensor.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            **kwargs: Additional arguments for base attention.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and attention weights.
        """
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.pool_g(k)

        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2] :, None])

        # Reshape for multi-head
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
        g = torch.logsigmoid(g) / gate_logit_normalizer

        q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))

        if self.use_rotary:
            print(
                "[LinearAttention] Applying rotary embedding (not implemented in this snippet)"
            )

        batch_size, num_heads, seq_len, head_dim = q.shape

        # Flatten for operator: [batch_size * num_heads * seq_len, head_dim]
        q_lin = q.reshape(batch_size * num_heads * seq_len, head_dim)
        k_lin = k.reshape(batch_size * num_heads * seq_len, head_dim)
        v_lin = v.reshape(batch_size * num_heads * seq_len, head_dim)
        g_lin = g.reshape(batch_size * num_heads * seq_len, head_dim)

        # Validate chunk size
        total_length = batch_size * num_heads * seq_len
        valid_chunk_size = get_valid_chunk_size(total_length, self.max_chunk_size)

        # Linear attention
        o_lin, _ = self.operator(
            q_lin, k_lin, v_lin, beta=g_lin, chunk_size=valid_chunk_size
        )
        o_lin = o_lin.reshape(batch_size, num_heads, seq_len, head_dim)
        o_lin = rearrange(o_lin, "b h n d -> b n (h d)")
        o_lin = self.o_proj(o_lin)

        # Standard attention
        o_base, attn_weights = self.base_attn(
            hidden_states, attention_mask=attention_mask, **kwargs
        )

        out = self.mag_weight * o_lin + (1 - self.mag_weight) * o_base
        return out, attn_weights
