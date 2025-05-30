import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    apply_linear_attention_mask,
    repeat_kv,
    truncate_attention_mask,
    split_qkv,
)

def apply_projections(hidden_states, base_attn):
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


def prepare_tensor_for_attn(hidden_states, attention_mask base_attn):
    # Apply projections to hidden states
    q, k, v, out_proj = apply_projections(hidden_states, self.base_attn)

    # Manage attention mask (with padding)
    if attention_mask is not None:
        # attention_mask -> [batch, seq], v: [batch, seq, ...]
        v = apply_linear_attention_mask(attention_mask, v)

    # Reshape for multi-head
    q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
    k = rearrange(k, "b n (h d) -> b h n d", h=self.num_key_value_heads)
    v = rearrange(v, "b n (h d) -> b h n d", h=self.num_key_value_heads)

    # Repeat for GQA
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

def linear_attention_forward(pool_g):
    # Gating
    g = pool_g(k)
    g = rearrange(g, "b n (h m) -> b h n m", h=self.num_key_value_heads)
    g = repeat_kv(g, self.num_key_value_groups)

    q = torch.clamp(F.softmax(q, dim=-1), min=1e-6, max=1 - 1e-6)
    k = torch.clamp(F.softmax(k, dim=-1), min=1e-6, max=1 - 1e-6)

    gate_norm = kwargs.get("gate_logit_normalizer", 16)
    g = F.logsigmoid(g) / gate_norm
    g = torch.clamp(g, min=-gate_norm, max=gate_norm)

    # Convert to float32 for numerical stability and get model dtype
    model_dtype = q.dtype
    q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))

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


def self_attention_forward():
    ### https://github.com/OpenSparseLLMs/Linearization/blob/main/liger/models/liger_gla/modeling_liger_gla.py
    # If cache_implementation="static" -> truncated attention
    if kwargs["use_cache"]:
        hidden_states, attention_mask = truncate_attention_mask(
            hidden_states, attention_mask, self.max_attn_length
        )
    # Standard attention (mask and rotation is applied inside)
    base_attn_outputs = self.base_attn(
        hidden_states, attention_mask=attention_mask, **kwargs
    )  # TODO : Get batch position for left_trunc ?
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rotary_emb=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary_emb = rotary_emb  # Callable or None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    ):
        # Projeter Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape pour multi-head [B, L, D] -> [B, H, L, D_head]
        B, L, _ = q.shape
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L, D_head]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Appliquer Rotary Embeddings si fourni
        if self.rotary_emb is not None:
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # À définir selon ton code

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask[:, None, None, :] == 0, float("-inf")
            )
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_probs, v)  # [B, H, L, D_head]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        )
        output = self.o_proj(attn_output)
        return output


# Utilisation dans un bloc Transformer
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, rotary_emb=None):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, rotary_emb=rotary_emb
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, x, attention_mask=None, position_ids=None, position_embeddings=None
    ):
        attn_out = self.self_attn(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        x = x + attn_out
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        return x


# Exemple d'utilisation
# rotary_emb = ... (ta fonction ou classe de rotary embedding)
# model = TransformerBlock(embed_dim=512, num_heads=8, rotary_emb=rotary_emb)
# output = model(hidden_states, attention_mask, position_ids, position_embeddings)
