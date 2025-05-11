from typing import List, Union, Callable, Optional, Dict
import torch.nn as nn
from .mpa import ParallelFLAAttention
from .config import AttentionConfig

class CustomInjectConfig:
    def __init__(
        self,
        target_modules: List[str],
        operator: Union[str, Callable] = "gla",
        combine_fn: Optional[Callable] = None,
        operator_kwargs: Optional[Dict] = None,
        fla_weight: float = 0.01,
    ):
        self.target_modules = target_modules
        self.operator = operator
        self.combine_fn = combine_fn
        self.operator_kwargs = operator_kwargs or {}
        self.fla_weight = fla_weight

    def custom_layer_factory(self, base_attn: nn.Module):
        config = AttentionConfig.parse_obj(base_attn.config.__dict__)
        return ParallelFLAAttention(
            base_attn,
            config,
            operator=self.operator,
            combine_fn=self.combine_fn,
            operator_kwargs=self.operator_kwargs,
            fla_weight=self.fla_weight
        )

def get_custom_injected_model(model: nn.Module, config: CustomInjectConfig) -> nn.Module:
    for name, module in model.named_modules():
        if name in config.target_modules:
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, config.custom_layer_factory(getattr(parent, last)))
    return model
