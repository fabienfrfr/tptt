"""Utilities to inject LiZA linear attention and manage mag_weight scheduling."""

from transformers import TrainerCallback

from .liza.memory_gate import LiZAttention


class AdjustMaGWeightCallback(TrainerCallback):
    """TrainerCallback to schedule mag_weight during training."""

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
            for _, module in self.model.named_modules():
                if isinstance(module, LiZAttention):
                    module.mag_weight = weight
