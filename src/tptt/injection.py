"""Utilities to inject LiZA linear attention and manage mag_weight scheduling."""

from torch import nn
from transformers.configuration_utils import PretrainedConfig

from .liza.memory_gate import LiZAttention
from .utils import LCache, extract_layer_idx


def inject_linear_attention(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    model: nn.Module,
    config: PretrainedConfig,  # ou LlamaConfig, MistralConfig, etc.
    liza_attention: LiZAttention,
    target_modules: list,
    linear_cache: LCache = None,
    operator_mode: str = "delta_rule",
    mag_weight: float = 0.5,
    max_chunk_size: int = 64,
):
    """Replace target modules in a model with LiZAttention."""
    linear_cache = linear_cache or LCache()
    # Inject LiZAttention into the model
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
                liza_attention(
                    getattr(parent, last),
                    layer_idx=layer_idx,
                    config=config,
                    linear_cache=linear_cache,
                    operator_mode=operator_mode,
                    mag_weight=mag_weight,
                    max_chunk_size=max_chunk_size,
                ),
            )
    return model, linear_cache
