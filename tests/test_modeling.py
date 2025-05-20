from unittest.mock import patch

from src.tptt.modeling_tptt import TpttModel


def test_tptt_config_default_values(dummy_tptt_config):
    config = dummy_tptt_config
    assert config.model_name == "test-model"
    assert "self_attn" in config.target_modules_names
    assert config.operator_mode == "delta_rule"
    assert config.mag_weight == 0.5
    assert config.max_chunk_size == 64


@patch(
    "src.tptt.modeling_tptt.AutoModelForCausalLM.from_pretrained"
)  # mocking the model
@patch("src.tptt.modeling_tptt.AutoTokenizer.from_pretrained")  # mocking the tokenizer
@patch("src.tptt.injection.inject_linear_attention")  # mocking the attention injection
def test_tpttmodel_init(
    mock_inject,
    mock_tokenizer,
    mock_model,
    dummy_model,
    dummy_tokenizer,
    cache,
    dummy_tptt_config,
):
    # Arrange: set up return values for the mocks
    mock_model.return_value = dummy_model
    mock_tokenizer.return_value = dummy_tokenizer
    mock_inject.return_value = (dummy_model, cache)

    # Act: instantiate the TpttModel
    tptt = TpttModel(dummy_tptt_config)

    # Assert
    assert tptt.config == dummy_tptt_config
    assert tptt.model is dummy_model
    assert tptt.tokenizer == dummy_tokenizer
