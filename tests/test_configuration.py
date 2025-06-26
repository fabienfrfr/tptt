from transformers import PretrainedConfig

from src.tptt.configuration_tptt import (
    convert_sets_to_lists,
    TpttConfig,
    extract_template_variables,
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
