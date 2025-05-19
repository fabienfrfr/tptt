from unittest.mock import patch

from src.tptt.model import TpttModel


def test_tptt_config_default_values(dummy_tptt_config):
    config = dummy_tptt_config
    assert config.model_name == "test-model"
    assert config.target_modules_names == "self_attn"
    assert config.operator_mode == "delta_rule"
    assert config.mag_weight == 0.5
    assert config.max_chunk_size == 64


@patch("src.tptt.model.AutoModelForCausalLM.from_pretrained")  # injecting the model
@patch("src.tptt.model.AutoTokenizer.from_pretrained")  # injecting the tokenizer
@patch("src.tptt.injection.inject_linear_attention")  # injecting the attention
def test_tpttmodel_init(
    mock_inject,
    mock_tokenizer,
    mock_model,
    dummy_model,
    dummy_tokenizer,
    cache,
    dummy_tptt_config,
):
    mock_model.return_value = dummy_model
    mock_tokenizer.return_value = dummy_tokenizer
    mock_inject.return_value = dummy_model

    tptt = TpttModel(model_name="dummy-model", config=dummy_tptt_config)
    assert tptt.model_name == "dummy-model"
    assert tptt.model[0] is dummy_model
    assert tptt.tokenizer == dummy_tokenizer
    assert "self_attn" in tptt.target_modules[0]
