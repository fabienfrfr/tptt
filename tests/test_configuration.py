import pytest

from transformers import PretrainedConfig

from src.tptt.configuration_tptt import (
    TpttConfig,
    convert_sets_to_lists,
    extract_template_variables,
    parse_mode_name,
    get_mode_name,
)


# Tests for convert_sets_to_lists
def test_convert_sets_to_lists_basic():
    input_data = {"a": {1, 2, 3}, "b": [4, 5, 6]}
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert convert_sets_to_lists(input_data) == expected


def test_convert_sets_to_lists_no_change():
    input_data = [1, 2, 3, "hello"]
    assert convert_sets_to_lists(input_data) == input_data


def test_tptt_config_merge_base_config():
    class MockConfig(PretrainedConfig):
        def __init__(self, hidden_size=128, **kwargs):
            self.hidden_size = hidden_size
            super().__init__(**kwargs)

    base_config = MockConfig(hidden_size=256)
    config = TpttConfig(base_model_config=base_config)
    assert config.hidden_size == 256


# Tests for extract_template_variables
def test_extract_template_variables_basic():
    template = "Hello {name}, your code is {code}"
    assert extract_template_variables(template) == {"name", "code"}


def test_extract_template_variables_nested():
    template = "{{outer}} {inner} {{nested{deep}}}"
    assert extract_template_variables(template) == {"outer", "inner", "deep"}


def test_extract_template_variables_empty():
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
    assert parse_mode_name(name) == expected_params
