import torch.nn as nn
from transformers import TrainerCallback
from transformers.configuration_utils import PretrainedConfig

from .linear_attention import LiZAttention


def inject_linear_attention(
    model: nn.Module,
    config: PretrainedConfig,  # ou LlamaConfig, LigerGLAConfig, etc.
    target_modules: list,
    operator_mode: str = "delta_rule",
    mag_weight: float = 0.5,
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
                LiZAttention(
                    getattr(parent, last),
                    config=config,
                    operator_mode=operator_mode,
                    mag_weight=mag_weight,
                    chunk_size=chunk_size,
                ),
            )
    return model


class AdjustMaGWeightCallback(TrainerCallback):
    def __init__(
        self, model, initial_weight=0.01, final_weight=0.5, transition_step=500
    ):
        self.model = model
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.transition_step = transition_step

    def on_step_end(self, args, state, control, **kwargs):
        # Calculate the current step in the transition phase
        current_step = state.global_step
        if current_step < self.transition_step:
            # Linear interpolation of the weight
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * (
                current_step / self.transition_step
            )
            for name, module in self.model.named_modules():
                if isinstance(module, LiZAttention):
                    module.mag_weight = weight
