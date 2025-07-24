# pylint: disable=too-many-arguments, too-many-positional-arguments

"""
Author : Fabien FURFARO
"""

from typing import Optional, Union

from transformers import PreTrainedModel, TrainerCallback

from .modeling_tptt import LiZAttention


class LiZACallback(TrainerCallback):
    """
    TrainerCallback to schedule mag_weight or enable/disable linear attention during training.

    Modes:
        - "gradual": linear interpolation from initial_weight to final_weight.
        - "cyclic": alternate between values in weight_list at each step.
        - "switch": alternately enable/disable linear attention at each step.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        mode: str = "gradual",
        initial_weight: float = 0.0,
        final_weight: float = 0.5,
        transition_step: Union[int, tuple, list] = 100,
        weight_list: Optional[list] = None,
        switch_period: int = 1,  # period for switching
    ):
        self.model = model
        self.mode = mode

        # Ensure initial_weight is a float scalar, not tuple/list
        if isinstance(initial_weight, (tuple, list)):
            initial_weight = initial_weight[0]
        if isinstance(final_weight, (tuple, list)):
            final_weight = final_weight[0]
        self.initial_weight = float(initial_weight)
        self.final_weight = float(final_weight)

        # Ensure transition_step is an int scalar, not tuple/list
        self.transition_step = ensure_int(transition_step)

        # For cyclic mode: ensure all weights are float scalars
        if weight_list is not None:
            self.weight_list = [
                float(w[0]) if isinstance(w, (tuple, list)) else float(w)
                for w in weight_list
            ]
        else:
            self.weight_list = [self.initial_weight, self.final_weight]

        # For switch_alternate mode
        self.switch_period = int(switch_period)

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        transition_step = self.transition_step

        # Ensure current_step and transition_step are plain ints
        current_step = ensure_int(current_step)
        transition_step = ensure_int(transition_step)

        # Select mag_weight or enable/disable linear attention according to mode
        if self.mode == "gradual":
            if current_step <= transition_step:
                weight = self.initial_weight + (
                    self.final_weight - self.initial_weight
                ) * (current_step / transition_step)
            else:
                weight = self.final_weight
            for _, module in self.model.named_modules():
                if isinstance(module, LiZAttention):
                    module.mag_weight = weight

        elif self.mode == "cyclic":
            idx = current_step % len(self.weight_list)
            weight = self.weight_list[idx]
            for _, module in self.model.named_modules():
                if isinstance(module, LiZAttention):
                    module.mag_weight = weight

        elif self.mode == "switch":
            # Alternately enable/disable linear attention every switch_period steps
            disable = (current_step // self.switch_period) % 2 == 0
            for _, module in self.model.named_modules():
                if isinstance(module, LiZAttention):
                    module.disable_linear_attn = disable

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        mag_weight = None
        disable_linear_attn = None
        # Log the current mag_weight and disable_linear_attn
        for _, module in self.model.named_modules():
            if isinstance(module, LiZAttention):
                mag_weight = getattr(module, "mag_weight", None)
                disable_linear_attn = getattr(module, "disable_linear_attn", None)
                break
        if mag_weight is not None and logs is not None:
            logs["mag_weight"] = float(mag_weight)
        if disable_linear_attn is not None and logs is not None:
            logs["disable_linear_attn"] = not bool(disable_linear_attn)


def ensure_int(value: Union[int, tuple, list]) -> int:
    """Ensure the value is a plain integer."""
    if isinstance(value, (tuple, list)):
        value = int(value[0])
    if hasattr(value, "item"):
        value = int(value.item())
    return value


class SaveBestModelCallback(TrainerCallback):
    """TrainerCallback to save the best model based on evaluation loss."""

    def __init__(self):
        self.best_metric = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            if metrics["eval_loss"] < self.best_metric:
                self.best_metric = metrics["eval_loss"]
                control.should_save = True  # Trigger save
            else:
                control.should_save = False  # Skip save
