import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from .linear_attention import LinearAttention


def inject_linear_attention(
    model: nn.Module,
    config: PretrainedConfig,  # ou LlamaConfig, LigerGLAConfig, etc.
    target_modules: list,
    operator_mode: str = "delta_rule",
    fla_weight: float = 0.5,
    chunk_size: int = 64,  # max
):
    for name, module in model.named_modules():
        if name in target_modules:
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(
                parent,
                last,
                LinearAttention(
                    getattr(parent, last),
                    config=config,
                    operator_mode=operator_mode,
                    fla_weight=fla_weight,
                    chunk_size=chunk_size,
                ),
            )
    return model
