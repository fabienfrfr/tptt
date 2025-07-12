"""
Author : Fabien FURFARO
"""

from transformers import TrainerCallback

from .modeling_tptt import LiZAttention


class AdjustMaGWeightCallback(TrainerCallback):
    """TrainerCallback to schedule mag_weight during training."""

    def __init__(
        self, model, initial_weight=0.01, final_weight=0.5, transition_step=500
    ):
        self.model = model
        # Ensure weights are always float scalars, not tuples/lists
        if isinstance(initial_weight, (tuple, list)):
            initial_weight = initial_weight[0]
        if isinstance(final_weight, (tuple, list)):
            final_weight = final_weight[0]
        self.initial_weight = float(initial_weight)
        self.final_weight = float(final_weight)

        if isinstance(transition_step, (tuple, list)):
            transition_step = transition_step[0]
        self.transition_step = int(transition_step)

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        transition_step = self.transition_step

        # Ensure both are plain ints (not tuple, list, tensor, numpy, etc.)
        if isinstance(current_step, (tuple, list)):
            current_step = current_step[0]
        if hasattr(current_step, "item"):
            current_step = int(current_step.item())
        else:
            current_step = int(current_step)

        if isinstance(transition_step, (tuple, list)):
            transition_step = transition_step[0]
        if hasattr(transition_step, "item"):
            transition_step = int(transition_step.item())
        else:
            transition_step = int(transition_step)

        if current_step <= transition_step:
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * (
                current_step / transition_step
            )
            for _, module in self.model.named_modules():
                if isinstance(module, LiZAttention):
                    module.mag_weight = weight

    def on_log(self, args, state, control, logs=None, **kwargs):
        mag_weight = None
        for _, module in self.model.named_modules():
            if isinstance(module, LiZAttention):
                mag_weight = getattr(module, "mag_weight", None)
                break
        if mag_weight is not None and logs is not None:
            logs["mag_weight"] = float(mag_weight)


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
