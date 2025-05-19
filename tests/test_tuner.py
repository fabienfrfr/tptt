from unittest.mock import MagicMock

from src.tptt.liza.memory_gate import LiZAttention
from src.tptt.tuner import AdjustMaGWeightCallback


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
