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
        # Ensure transition_step is always an int, even if a tuple/list is passed (e.g., with DDP)
        if isinstance(transition_step, (tuple, list)):
            transition_step = transition_step[0]
        self.transition_step = int(transition_step)

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        # Safety: in case transition_step somehow becomes a tuple again
        transition_step = self.transition_step
        if isinstance(transition_step, (tuple, list)):
            transition_step = transition_step[0]
        # Linearly interpolate mag_weight during the transition phase
        if current_step < transition_step:
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * (
                current_step / transition_step
            )
            # Update mag_weight for all LiZAttention modules in the model
            for _, module in self.model.named_modules():
                if isinstance(module, LiZAttention):
                    module.mag_weight = weight
