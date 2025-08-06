# pylint: disable=too-few-public-methods, too-many-branches
"""Tests config for the tppt module."""

from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

import src.tptt.configuration_tptt as tpttconf
from src.tptt.configuration_tptt import (TpttConfig, convert_sets_to_lists,
                                         extract_template_variables,
                                         get_mode_name, parse_mode_name)


def test_tptt_config_base_model_is_none(monkeypatch):  # pylint: disable=unused-argument
    """Test that config loads from AutoConfig.from_pretrained if base_model_config is None."""
    # Patch from_pretrained
    with patch("src.tptt.configuration_tptt.AutoConfig.from_pretrained") as mock_auto:
        dummy_conf = {"hidden_size": 77, "someparam": 42}
        mock_auto.return_value.to_dict.return_value = dummy_conf
        conf = TpttConfig(base_model_config=None, base_model_name="test/model_a")
        assert conf.hidden_size == 77
        assert conf.base_model_name == "test/model_a"


def test_tptt_config_linear_precision_dtype():
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


def test_tptt_config_lora_peft_type_enum():
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


def test_parse_mode_name_invalid_mode():
    """Test that parse_mode_name raises for unknown mode."""
    with pytest.raises(ValueError):
        parse_mode_name("unknown_super_mode")


def test_generate_model_card_basic(tmp_path, monkeypatch):
    """Test generate_model_card writes variables to README.md."""
    # Crée un template simple avec 2 champs
    template_txt = "Hello {model_id} and {hidden_size}!"
    template_path = tmp_path / "model_card_template.md"
    template_path.write_text(template_txt, encoding="utf-8")

    # Patch __file__ pour os.path.dirname(__file__)
    monkeypatch.setattr(tpttconf, "__file__", str(template_path))

    class DummyConfig:
        """Dummy config"""

        def __init__(self):
            self.__dict__ = {"hidden_size": 123}

    # Appel de generate_model_card
    tpttconf.generate_model_card(str(tmp_path), DummyConfig())

    # Vérification du README généré
    out = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Hello" in out and "123" in out and "README.md" not in out


def test_generate_model_card_flatten_dict(tmp_path, monkeypatch):
    """Test flatten_config for dict and non-dict values."""
    template_txt = "{dummy_foo_bar} - {dummy_baz}"
    template_path = tmp_path / "model_card_template.md"
    template_path.write_text(template_txt, encoding="utf-8")

    monkeypatch.setattr(tpttconf, "__file__", str(template_path))

    class DummyConfig:  # pylint: disable=too-few-public-methods
        """Dummy config"""

        def __init__(self):
            self.dummy_foo = {"bar": 42}
            self.dummy_baz = 99
            self.__dict__ = {"dummy_foo": {"bar": 42}, "dummy_baz": 99}

    tpttconf.generate_model_card(str(tmp_path), DummyConfig())

    out = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "42" in out and "99" in out


def test_generate_model_card_missing_variable(tmp_path, monkeypatch):
    """Test that missing variables are filled as 'N/A'."""
    template_txt = "Hello {present} & {missing} !"
    template_path = tmp_path / "model_card_template.md"
    template_path.write_text(template_txt, encoding="utf-8")

    monkeypatch.setattr(tpttconf, "__file__", str(template_path))

    class DummyConfig:  # pylint: disable=too-few-public-methods
        """Dummy conf"""

        def __init__(self):
            self.present = "Here"
            self.__dict__ = {"present": "Here"}

    tpttconf.generate_model_card(str(tmp_path), DummyConfig())
    out = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Here" in out
    assert "N/A" in out  # pour {missing}


def test_generate_model_card_list_via_kwargs(tmp_path, monkeypatch):
    """Directly pass a list in kwargs to ensure variables[k] is a list."""
    template_txt = "K: {k}"
    template_path = tmp_path / "model_card_template.md"
    template_path.write_text(template_txt, encoding="utf-8")

    monkeypatch.setattr(tpttconf, "__file__", str(template_path))

    class DummyConfig:  # pylint: disable=too-few-public-methods
        """Dummy config"""  # no "k" in dict

        def __init__(self):
            self.__dict__ = {}

    tpttconf.generate_model_card(str(tmp_path), DummyConfig(), k=[1, 2, 3])
    out = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "1, 2, 3" in out


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
def test_tptt_config_bidirectional_true(capfd):
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


def test_tptt_config_padding_side_default(caplog):
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


def test_tptt_config_custom_operator_mode():
    """test custom config"""

    class DummyConfig(PretrainedConfig):
        """Dummy config, change to conftest.py dummy"""

        def __init__(self):

            super().__init__()
            self.hidden_size = 64

    config = TpttConfig(
        base_model_config=DummyConfig(), operator_mode="delta_product_kv_gelu_r"
    )
    assert config.recurrent_config["trick"] == "rotative"
    assert config.recurrent_config["gate_type"] == "kv"


def test_extract_template_variables_basic():
    """Tests for basic extract_template_variables"""
    template = "Hello {name}, your code is {code}"
    assert extract_template_variables(template) == {"name", "code"}


def test_extract_template_variables_nested():
    """Tests for nested extract_template_variables"""
    template = "{{outer}} {inner} {{nested{deep}}}"
    assert extract_template_variables(template) == {"outer", "inner", "deep"}


def test_extract_template_variables_empty():
    """Tests for empty extract_template_variables"""
    assert extract_template_variables("No variables") == set()


@pytest.mark.parametrize(
    "params,expected_name",
    [
        ((1, "k", True, "derivative"), "delta_rule"),
        ((1, "v", True, "derivative"), "delta_rule_v"),
        ((1, "kv", False, "derivative"), "delta_rule_kv_gelu"),
        ((2, "k", True, "derivative"), "delta_product"),
        ((2, "k", True, "rotative"), "delta_product_r"),
        ((2, "k", True, "combined"), "delta_product_c"),
        ((2, "kv", False, "derivative"), "delta_product_kv_gelu"),
        ((2, "kv", False, "rotative"), "delta_product_kv_gelu_r"),
        ((3, "kv", False, "rotative"), "delta_product_3_kv_gelu_r"),
        ((3, "v", False, "combined"), "delta_product_3_v_gelu_c"),
    ],
)
def test_get_mode_name(params, expected_name):
    """Test param from from mode name"""
    assert get_mode_name(*params) == expected_name


@pytest.mark.parametrize(
    "name,expected_params",
    [
        (
            "delta_rule",
            {"order": 1, "gate_type": "k", "linear": True, "trick": "derivative"},
        ),
        (
            "delta_rule_v",
            {"order": 1, "gate_type": "v", "linear": True, "trick": "derivative"},
        ),
        (
            "delta_rule_kv_gelu",
            {"order": 1, "gate_type": "kv", "linear": False, "trick": "derivative"},
        ),
        (
            "delta_product",
            {"order": 2, "gate_type": "k", "linear": True, "trick": "derivative"},
        ),
        (
            "delta_product_r",
            {"order": 2, "gate_type": "k", "linear": True, "trick": "rotative"},
        ),
        (
            "delta_product_c",
            {"order": 2, "gate_type": "k", "linear": True, "trick": "combined"},
        ),
        (
            "delta_product_kv_gelu",
            {"order": 2, "gate_type": "kv", "linear": False, "trick": "derivative"},
        ),
        (
            "delta_product_kv_gelu_r",
            {"order": 2, "gate_type": "kv", "linear": False, "trick": "rotative"},
        ),
        (
            "delta_product_3_v_gelu_c",
            {"order": 3, "gate_type": "v", "linear": False, "trick": "combined"},
        ),
    ],
)
def test_parse_mode_name(name, expected_params):
    """Test get mode name from param"""
    assert parse_mode_name(name) == expected_params
