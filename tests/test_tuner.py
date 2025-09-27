"""Tests tuning LiZA for the tppt module."""

from unittest.mock import MagicMock

import pytest

from src.tptt.modeling_tptt import LiZAttention
from src.tptt.train_tptt import LiZACallback, SaveBestModelCallback, ensure_int


class DummyMemoryGate:
    def __init__(self):
        self._mag_weight = 0.5

    @property
    def mag_weight(self):
        if hasattr(self._mag_weight, "item"):
            return float(self._mag_weight.item())
        return self._mag_weight

    @mag_weight.setter
    def mag_weight(self, value):
        if hasattr(value, "item"):
            value = float(value.item())
        self._mag_weight = value


class DummyLiZAttention:
    def __init__(self):
        self.memory_gate = DummyMemoryGate()
        self.disable_linear_attn = False

    @property
    def mag_weight(self):
        return self.memory_gate.mag_weight

    @mag_weight.setter
    def mag_weight(self, value):
        self.memory_gate.mag_weight = value


class DummyModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return [(f"liz{i}", module) for i, module in enumerate(self._modules)]


@pytest.mark.parametrize(
    "global_step, expected_disable, msg",
    [
        (0, True, "disable_linear_attn should be True at step 0"),
        (1, True, "disable_linear_attn should be True at step 1"),
        (2, False, "disable_linear_attn should be False at step 2"),
        (3, False, "disable_linear_attn should be False at step 3"),
        (4, True, "disable_linear_attn should be True at step 4"),
        (5, True, "disable_linear_attn should be True at step 5"),
    ],
)
def test_lizacallback_switch_mode_param(global_step, expected_disable, msg):
    """Test LiZACallback switch mode using parametrize."""
    liz = MagicMock(spec=LiZAttention)
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]
    callback = LiZACallback(model, mode="switch", switch_period=2)
    state = MagicMock()
    state.global_step = global_step
    callback.on_step_end(None, state, None)
    assert liz.disable_linear_attn is expected_disable, msg


def test_callback_updates_multiple_modules():
    "Test callback update during training with multiple LiZAttention modules."
    liz1 = DummyLiZAttention()
    liz2 = DummyLiZAttention()
    model = DummyModel([liz1, liz2])

    callback = LiZACallback(model, transition_step=100)

    class DummyState:
        global_step = 50

    state = DummyState()

    callback.on_step_end(None, state, None)

    expected = 0.0 + (0.5 - 0.0) * (50 / 100)
    assert abs(liz1.mag_weight - expected) < 1e-6
    assert abs(liz2.mag_weight - expected) < 1e-6


def test_adjust_mag_weight_callback_simple():
    "Test mag_weight adjustment by callback during training."
    liz = DummyLiZAttention()
    model = DummyModel([liz])

    callback = LiZACallback(
        model, initial_weight=0.1, final_weight=0.5, transition_step=100
    )

    class DummyState:
        global_step = 50

    state = DummyState()

    callback.on_step_end(None, state, None)

    expected = 0.1 + (0.5 - 0.1) * (50 / 100)
    assert abs(liz.mag_weight - expected) < 1e-6


def test_callback_handles_tuple_parameters():
    "Test callback parameters as tuples are handled correctly."
    liz = DummyLiZAttention()
    model = DummyModel([liz])

    callback = LiZACallback(
        model, initial_weight=(0.1,), final_weight=(0.5,), transition_step=(100,)
    )

    class DummyState:
        global_step = 50

    state = DummyState()

    callback.on_step_end(None, state, None)

    expected = 0.1 + (0.5 - 0.1) * (50 / 100)
    assert abs(liz.mag_weight - expected) < 1e-6


def test_callback_handles_tensor_steps():
    "Test callback handles PyTorch tensor global_step."
    liz = DummyLiZAttention()
    model = DummyModel([liz])

    callback = LiZACallback(model, transition_step=100)

    class DummyTensorStep:
        def item(self):
            return 50

    class DummyState:
        global_step = DummyTensorStep()

    state = DummyState()

    callback.on_step_end(None, state, None)

    expected = 0.0 + (0.5 - 0.0) * (50 / 100)
    assert abs(liz.mag_weight - expected) < 1e-6


def test_gradual_sets_final_weight_after_transition():
    "Test mag_weight equals final_weight after transition step."
    liz = DummyLiZAttention()
    model = DummyModel([liz])
    callback = LiZACallback(
        model, initial_weight=0.2, final_weight=0.6, transition_step=10
    )

    class DummyState:
        global_step = 15

    state = DummyState()

    callback.on_step_end(None, state, None)

    assert abs(liz.mag_weight - 0.6) < 1e-6


def test_lizacallback_cyclic_mode():
    "Test cyclic mode cycles through weights in callback."
    liz = DummyLiZAttention()
    model = DummyModel([liz])

    weights = [0.3, 0.7, 0.5]
    callback = LiZACallback(model, mode="cyclic", weight_list=weights)

    class DummyState:
        global_step = 0

    state = DummyState()

    for i in range(len(weights) * 2):  # 2 cycles
        state.global_step = i
        callback.on_step_end(None, state, None)
        assert abs(liz.mag_weight - weights[i % len(weights)]) < 1e-6


def test_lizacallback_raises_on_unknown_mode():
    """Test LiZACallback raises ValueError for unknown mode."""
    model = MagicMock()
    liz = MagicMock(spec=LiZAttention)
    model.named_modules.return_value = [("liz", liz)]
    callback = LiZACallback(model, mode="badmode")
    state = MagicMock()
    state.global_step = 0
    with pytest.raises(ValueError):
        callback.on_step_end(None, state, None)


def test_on_log_no_liza_module():
    """Test on_log does nothing if no LiZAttention module found."""
    model = MagicMock()
    model.named_modules.return_value = [("notliz", MagicMock())]
    callback = LiZACallback(model)
    logs = {}
    callback.on_log(None, None, None, logs)
    # Should not add anything to logs
    assert not logs


def test_on_log_disable_linear_attn_logged():
    """Test on_log logs disable_linear_attn if present."""
    liz = MagicMock(spec=LiZAttention)
    liz.disable_linear_attn = True
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]
    callback = LiZACallback(model)

    logs = {}
    callback.on_log(None, None, None, logs)
    # Should log the inverse (not True => False) --> fixed
    assert logs["disable_linear_attn"] is True


def test_ensure_int_handles_tensor_like():
    """Test ensure_int with tensors having item() method."""

    class DummyTensor:  # pylint: disable=too-few-public-methods
        """23 dummyy"""

        def item(self):
            """dummy item"""
            return 23

    assert ensure_int(DummyTensor()) == 23
    # Also check tuple/list extraction
    assert ensure_int([5]) == 5
    assert ensure_int((10,)) == 10
    # And direct integer
    assert ensure_int(42) == 42


def test_callback_logs_mag_weight():
    """Test monitoring mag logs"""
    liz = MagicMock(spec=LiZAttention)
    liz.mag_weight = 0.3
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]

    callback = LiZACallback(model)
    logs = {}

    callback.on_log(None, None, None, logs)

    assert logs["mag_weight"] == 0.3


@pytest.mark.parametrize(
    "previous_best, eval_loss, expected_best, expected_should_save",
    [
        (float("inf"), 0.8, 0.8, True),
        (0.8, 0.9, 0.8, False),  # no update
        (0.8, 0.7, 0.7, True),  # new update
    ],
)
def test_save_best_model_callback_parametrized(
    previous_best, eval_loss, expected_best, expected_should_save
):
    """Test best saving model (optional class)"""
    callback = SaveBestModelCallback()
    callback.best_metric = previous_best
    args, state = MagicMock(), MagicMock()
    control = MagicMock()
    metrics = {"eval_loss": eval_loss}

    callback.on_evaluate(args, state, control, metrics)
    assert callback.best_metric == expected_best
    assert control.should_save is expected_should_save
