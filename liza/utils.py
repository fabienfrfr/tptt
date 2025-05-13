import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    return x.repeat_interleave(n_rep, dim=1)


def match_dim(x, dim, target_size):
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
        x = F.linear(x, eye)
    x = torch.moveaxis(x, -1, dim)
    return x


def get_valid_chunk_size(total_L, chunk_size):
    """
    Retourne le plus grand chunk_size <= chunk_size qui divise total_L.
    Si aucun chunk_size > 1 ne convient, retourne 1.
    """
    for c in range(min(chunk_size, total_L), 0, -1):
        if total_L % c == 0:
            return c
    return 1
