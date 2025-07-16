from unittest.mock import MagicMock

import pytest

from src.tptt.modeling_tptt import LiZAttention
from src.tptt.train_tptt import LiZACallback, SaveBestModelCallback


def test_adjust_mag_weight_callback_simple():
    liz = MagicMock(spec=LiZAttention)
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]

    callback = LiZACallback(
        model, initial_weight=0.1, final_weight=0.5, transition_step=100
    )
    state = MagicMock()
    state.global_step = 50

    callback.on_step_end(None, state, None)

    expected = 0.1 + (0.5 - 0.1) * (50 / 100)
    assert liz.mag_weight == expected


def test_callback_handles_tuple_parameters():
    liz = MagicMock(spec=LiZAttention)
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]

    # Test avec paramètres sous forme de tuples
    callback = LiZACallback(
        model, initial_weight=(0.1,), final_weight=(0.5,), transition_step=(100,)
    )

    state = MagicMock(global_step=50)
    callback.on_step_end(None, state, None)

    expected = 0.1 + (0.5 - 0.1) * (50 / 100)
    assert liz.mag_weight == expected


def test_callback_handles_tensor_steps():
    liz = MagicMock(spec=LiZAttention)
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]

    callback = LiZACallback(model, transition_step=100)

    # Simule un tensor PyTorch
    state = MagicMock()
    state.global_step = MagicMock()
    state.global_step.item.return_value = 50

    callback.on_step_end(None, state, None)

    expected = 0.0 + (0.5 - 0.0) * (50 / 100)
    assert liz.mag_weight == expected


def test_callback_logs_mag_weight():
    liz = MagicMock(spec=LiZAttention)
    liz.mag_weight = 0.3
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]

    callback = LiZACallback(model)
    logs = {}

    callback.on_log(None, None, None, logs)

    assert logs["mag_weight"] == 0.3


def test_callback_updates_multiple_modules():
    liz1 = MagicMock(spec=LiZAttention)
    liz2 = MagicMock(spec=LiZAttention)
    model = MagicMock()
    model.named_modules.return_value = [("liz1", liz1), ("liz2", liz2)]

    callback = LiZACallback(model, transition_step=100)
    state = MagicMock(global_step=50)

    callback.on_step_end(None, state, None)

    expected = 0.0 + (0.5 - 0.0) * (50 / 100)
    assert liz1.mag_weight == expected
    assert liz2.mag_weight == expected


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
    callback = SaveBestModelCallback()
    callback.best_metric = previous_best
    args, state = MagicMock(), MagicMock()
    control = MagicMock()
    metrics = {"eval_loss": eval_loss}

    callback.on_evaluate(args, state, control, metrics)
    assert callback.best_metric == expected_best
    assert control.should_save is expected_should_save
