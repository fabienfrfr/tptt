# pylint: disable=too-few-public-methods, too-many-branches
"""Tests config for the tppt module."""

from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

import src.tptt.configuration_tptt as tpttconf
from src.tptt.configuration_tptt import (
    TpttConfig,
    convert_sets_to_lists,
    get_model_name,
)


def test_liza_config_base_model_is_none(monkeypatch):  # pylint: disable=unused-argument
    """Test that config loads from AutoConfig.from_pretrained if base_model_config is None."""
    # Patch from_pretrained
    with patch("src.tptt.configuration_tptt.AutoConfig.from_pretrained") as mock_auto:
        dummy_conf = {"hidden_size": 77, "someparam": 42}
        mock_auto.return_value.to_dict.return_value = dummy_conf
        conf = TpttConfig(base_model_config=None, base_model_name="test/model_a")
        assert conf.hidden_size == 77
        assert conf.base_model_name == "test/model_a"


def test_liza_config_linear_precision_dtype():
    """Test config casts torch dtype linear_precision to string."""

    class DummyConfig(PretrainedConfig):
        """Dummy config"""

        def __init__(self):
            super().__init__()
            self.hidden_size = 128

    conf = TpttConfig(base_model_config=DummyConfig(), linear_precision=torch.float16)
    assert conf.linear_precision == "float16"


class DummyEnum:  # pylint: disable=too-few-public-methods
    """Dummy enumerator"""

    value = "dummy_enum_val"


def test_liza_config_lora_peft_type_enum():
    """Test lora_config with peft_type as enum-like object."""

    class DummyConfig(PretrainedConfig):
        """Dummy pretrained config"""

        def __init__(self):
            super().__init__()
            self.hidden_size = 16

    lora_conf = {"peft_type": DummyEnum(), "foo": {1, 2}}
    conf = TpttConfig(base_model_config=DummyConfig(), lora_config=lora_conf)
    assert conf.lora_config["peft_type"] == "dummy_enum_val"
    assert isinstance(conf.lora_config["foo"], list)


def test_generate_model_card_with_path(tmp_path):
    """It should generate a README.md using a direct template file path."""
    # Create a simple Jinja2 template with one config variable
    template_content = "Hello {{ model_id }} and {{ config.hidden_size }}!"
    template_path = tmp_path / "custom_template.md"
    template_path.write_text(template_content, encoding="utf-8")

    # Dummy config
    class DummyConfig:
        def __init__(self):
            self.hidden_size = 123

    # Call function with direct template path
    tpttconf.generate_model_card(
        output_path=str(tmp_path), config=DummyConfig(), template=str(template_path)
    )

    # Verify README was generated correctly
    readme_path = tmp_path / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert "Hello" in content
    assert "123" in content


def test_generate_model_card_with_template_name(tmp_path, monkeypatch):
    """It should load a template by name from the default templates directory."""
    # Create a fake templates folder and monkeypatch __file__
    fake_templates_dir = tmp_path / "templates"
    fake_templates_dir.mkdir()
    template_content = "Name: {{ model_id }}, Size: {{ config.size }}"
    template_file = fake_templates_dir / "model_card_template.md"
    template_file.write_text(template_content, encoding="utf-8")

    monkeypatch.setattr(tpttconf, "__file__", str(tmp_path / "dummy_module.py"))

    class DummyConfig:
        def __init__(self):
            self.size = 42

    tpttconf.generate_model_card(
        output_path=str(tmp_path),
        config=DummyConfig(),
        template=None,  # Will default to "model_card_template"
    )

    content = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "42" in content


def test_generate_model_card_extra_variables(tmp_path):
    """It should accept and render extra_variables dict from the user."""
    template_content = "Extra: {{ extra }}, Config: {{ config.value }}"
    template_path = tmp_path / "template.md"
    template_path.write_text(template_content, encoding="utf-8")

    class DummyConfig:
        def __init__(self):
            self.value = "cfg"

    tpttconf.generate_model_card(
        output_path=str(tmp_path),
        config=DummyConfig(),
        template=str(template_path),
        extra_variables={"extra": "custom_value"},
    )

    output = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "custom_value" in output
    assert "cfg" in output


def test_generate_model_card_missing_template_raises(tmp_path):
    """It should raise FileNotFoundError if template does not exist."""

    class DummyConfig:
        pass

    with pytest.raises(FileNotFoundError):
        tpttconf.generate_model_card(
            output_path=str(tmp_path),
            config=DummyConfig(),
            template="nonexistent_template_name",
        )


def test_convert_sets_to_lists_basic():
    """Tests for convert_sets_to_lists"""
    input_data = {"a": {1, 2, 3}, "b": [4, 5, 6]}
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert convert_sets_to_lists(input_data) == expected


def test_convert_sets_to_lists_no_change():
    """Test convert sets to list, used for LoRA serialization"""
    input_data = [1, 2, 3, "hello"]
    assert convert_sets_to_lists(input_data) == input_data


# capfd capturing standard output (standart fixture for pytest)
def test_liza_config_bidirectional_true(capfd):
    """Test bidirectional config test"""

    class DummyConfig(PretrainedConfig):
        """Dummy config, change to conftest.py dummy"""

        def __init__(self):
            super().__init__()
            self.hidden_size = 256

    config = TpttConfig(base_model_config=DummyConfig(), bidirectional=True)
    out, _ = capfd.readouterr()
    assert "Bidirectional is enabled" in out
    assert config.bidirectional is True


def test_liza_config_padding_side_default(caplog):
    """Test padding config"""

    class DummyConfig(PretrainedConfig):
        """Dummy config, change to conftest.py dummy"""

        def __init__(self):
            super().__init__()
            self.hidden_size = 128

    with caplog.at_level("INFO"):
        config = TpttConfig(base_model_config=DummyConfig(), padding_side=None)
    assert any("defaulting to 'right'" in message for message in caplog.messages)
    assert config.padding_side == "right"


@pytest.mark.parametrize(
    "params,expected_name",
    [
        # lora, order, alpha_gate, beta_gate, linear, trick
        (
            (True, False, False, 1, "c", "k", True, "dt"),
            "liza_lora_mag_causal_alpha-c_beta-k_order-1_linear_trick-dt",
        ),
        (
            (True, False, False, 1, "c", "v", True, "dt"),
            "liza_lora_mag_causal_alpha-c_beta-v_order-1_linear_trick-dt",
        ),
        (
            (True, False, False, 1, "c", "kv", False, "dt"),
            "liza_lora_mag_causal_alpha-c_beta-kv_order-1_gelu_trick-dt",
        ),
        (
            (True, False, False, 2, "c", "k", True, "dt"),
            "liza_lora_mag_causal_alpha-c_beta-k_order-2_linear_trick-dt",
        ),
        (
            (True, False, False, 2, "c", "k", True, "rot"),
            "liza_lora_mag_causal_alpha-c_beta-k_order-2_linear_trick-rot",
        ),
        (
            (True, False, False, 2, "c", "k", True, "rdt"),
            "liza_lora_mag_causal_alpha-c_beta-k_order-2_linear_trick-rdt",
        ),
        (
            (True, False, False, 2, "c", "kv", False, "dt"),
            "liza_lora_mag_causal_alpha-c_beta-kv_order-2_gelu_trick-dt",
        ),
        (
            (True, False, False, 2, "c", "kv", False, "rot"),
            "liza_lora_mag_causal_alpha-c_beta-kv_order-2_gelu_trick-rot",
        ),
        (
            (True, False, False, 3, "c", "kv", False, "rot"),
            "liza_lora_mag_causal_alpha-c_beta-kv_order-3_gelu_trick-rot",
        ),
        (
            (True, False, False, 3, "c", "v", False, "rdt"),
            "liza_lora_mag_causal_alpha-c_beta-v_order-3_gelu_trick-rdt",
        ),
    ],
)
def test_get_model_name(params, expected_name):
    # Disable date since it changes every day
    result = get_model_name(*params, prefix="liza", add_date=False)
    assert result == expected_name
