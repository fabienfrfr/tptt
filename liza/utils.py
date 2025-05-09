import torch
from einops import repeat

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    return repeat(x, 'b h n d -> b (h r) n d', r=n_rep)
