from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file

from src.tptt.modeling_tptt import load_tptt_safetensors, save_tptt_safetensors

MODULE = "src.tptt.modeling_tptt"  # for patch


@pytest.mark.parametrize(
    "params_requires_grad,expected_keys",
    [
        ([False, False], []),
        ([True, False], ["base.weight", "base.bias"]),
        # ([False, True],  ["A.B.weight", "A.B.bias"],),
    ],
)
def test_save_tptt_safetensor_peft(
    params_requires_grad, expected_keys, tmp_path, monkeypatch
):
    # Prepare model
    class Toy(torch.nn.Module):
        def __init__(self, req_grad):
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.lora = torch.nn.Linear(2, 2)
            self.base.weight.requires_grad_(req_grad[0])
            self.base.bias.requires_grad_(req_grad[0])

            self.lora.weight.requires_grad_(req_grad[1])
            self.lora.bias.requires_grad_(req_grad[1])

        def state_dict(self, *a, **k):
            sd = super().state_dict()
            # simulate LiZAttention key for lora
            sd["A.base_attn.B.weight"] = sd.pop("lora.weight")
            sd["A.base_attn.B.bias"] = sd.pop("lora.bias")
            return sd

        def named_parameters(self, *a, **k):
            for name, p in super().named_parameters():
                if name == "lora.weight":
                    yield "A.base_attn.B.weight", p
                elif name == "lora.bias":
                    yield "A.base_attn.B.bias", p
                else:
                    yield name, p

    model = Toy(params_requires_grad)
    file_path = tmp_path / "adapter_model.safetensors"
    # Save
    save_tptt_safetensors(model, tmp_path, "adapter_model.safetensors")

    # Load back and check
    if not expected_keys:
        assert not file_path.exists() or len(load_file(str(file_path)).keys()) == 0
    else:
        loaded = load_file(str(file_path))
        # adaptation: trucA.LiZAttention.trucB.weight -> trucA.trucB.weight
        assert set(loaded.keys()) == set(expected_keys)


@pytest.mark.parametrize(
    "model_keys, fake_sd, expect_missing, expect_unexpected",
    [
        # 1. Warning ONLY missing_lora
        # (["a.lora_A.default.weight", "foo"], {"foo": "bar"}, True, False),
        # 2. Warning ONLY unexpected
        (["foo"], {"foo": "bar", "surprise": "baz"}, False, True),
        # 3. Warning on both
        (["a.lora_A.default.weight"], {"surprise": "baz"}, True, True),
    ],
)
def test_load_tptt_safetensors_warn_branches(
    monkeypatch, model_keys, fake_sd, expect_missing, expect_unexpected
):
    """Test that logger.warning triggers for missing_lora and/or unexpected keys as needed."""

    def make_fake_model(state_keys):
        """make faker"""
        model = MagicMock()
        model.state_dict.return_value = {k: None for k in state_keys}

        def load_state_dict(sd, strict, assign):
            missing = [k for k in model.state_dict() if k not in sd]
            unexpected = [k for k in sd if k not in model.state_dict()]
            return missing, unexpected

        model.load_state_dict.side_effect = load_state_dict
        return model

    model = make_fake_model(model_keys)

    monkeypatch.setattr(f"{MODULE}.os.path.isdir", lambda x: True)
    monkeypatch.setattr(f"{MODULE}.os.path.exists", lambda x: True)
    monkeypatch.setattr(f"{MODULE}.os.path.join", lambda a, b: f"{a}/{b}")

    safe_open_ctx = MagicMock()
    safe_open_ctx.__enter__.return_value.keys.return_value = list(fake_sd)
    safe_open_ctx.__enter__.return_value.get_tensor.side_effect = lambda k: fake_sd[k]
    monkeypatch.setattr(f"{MODULE}.safe_open", MagicMock(return_value=safe_open_ctx))

    logger_mock = MagicMock()
    monkeypatch.setattr(f"{MODULE}.logger", logger_mock)

    load_tptt_safetensors("/fakepath", model)

    # Check warnings
    calls = [c[0][0] for c in logger_mock.warning.call_args_list]
    assert ("Missing LoRA keys" in "".join(calls)) is expect_missing
    assert ("Unexpected keys" in "".join(calls)) is expect_unexpected
