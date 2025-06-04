from unittest.mock import MagicMock

from src.tptt.modeling_tptt import LiZAttention
from src.tptt.train_tptt import AdjustMaGWeightCallback


def test_adjust_mag_weight_callback_simple():
    liz = MagicMock(spec=LiZAttention)
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]

    callback = AdjustMaGWeightCallback(
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
    callback = AdjustMaGWeightCallback(
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

    callback = AdjustMaGWeightCallback(model, transition_step=100)

    # Simule un tensor PyTorch
    state = MagicMock()
    state.global_step = MagicMock()
    state.global_step.item.return_value = 50

    callback.on_step_end(None, state, None)

    expected = 0.01 + (0.5 - 0.01) * (50 / 100)
    assert liz.mag_weight == expected


def test_callback_logs_mag_weight():
    liz = MagicMock(spec=LiZAttention)
    liz.mag_weight = 0.3
    model = MagicMock()
    model.named_modules.return_value = [("liz", liz)]

    callback = AdjustMaGWeightCallback(model)
    logs = {}

    callback.on_log(None, None, None, logs)

    assert logs["mag_weight"] == 0.3


def test_callback_updates_multiple_modules():
    liz1 = MagicMock(spec=LiZAttention)
    liz2 = MagicMock(spec=LiZAttention)
    model = MagicMock()
    model.named_modules.return_value = [("liz1", liz1), ("liz2", liz2)]

    callback = AdjustMaGWeightCallback(model, transition_step=100)
    state = MagicMock(global_step=50)

    callback.on_step_end(None, state, None)

    expected = 0.01 + (0.5 - 0.01) * (50 / 100)
    assert liz1.mag_weight == expected
    assert liz2.mag_weight == expected
