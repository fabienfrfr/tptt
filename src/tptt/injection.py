"""Utilities to inject LiZA linear attention and manage mag_weight scheduling."""

from torch import nn
from transformers.configuration_utils import PretrainedConfig

from .liza.memory_gate import LiZAttention
from .utils import extract_layer_idx


def inject_linear_attention(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    model: nn.Module,
    config: PretrainedConfig,  # ou LlamaConfig, LigerGLAConfig, etc.
    target_modules: list,
    operator_mode: str = "delta_rule",
    mag_weight: float = 0.5,
    max_chunk_size: int = 64,
):
    """Replace target modules in a model with LiZAttention."""
    for name, _ in model.named_modules():
        if name in target_modules:
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            layer_idx = extract_layer_idx(name)
            setattr(
                parent,
                last,
                LiZAttention(
                    getattr(parent, last),
                    layer_idx=layer_idx,
                    config=config,
                    operator_mode=operator_mode,
                    mag_weight=mag_weight,
                    max_chunk_size=max_chunk_size,
                ),
            )
    return model
