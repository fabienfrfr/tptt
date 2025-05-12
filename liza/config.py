from typing import Optional

from pydantic import BaseModel


class AttentionConfig(BaseModel):
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: Optional[int] = None
