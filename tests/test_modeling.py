"""Unit tests for the TpttModel and related classes."""

from unittest.mock import MagicMock  # , patch

import torch

# from src.tptt.modeling_tptt import TpttModel
from src.tptt.pipeline_tptt import TpttPipeline

# Patching is not used, high RAM usage in CI...
# @patch("transformers.AutoConfig.from_pretrained")
# def test_tptt_config_default_values(dummy_tptt_config):
#     """Test default values of TpttConfig."""
#     dummy_tptt_config.return_value.to_dict.return_value = {"model_name": "test-model"}
#     config = dummy_tptt_config
#     assert config.model_name == "test-model"
#     assert "self_attn" in config.target_modules_names
#     assert config.operator_mode == "delta_rule"
#     assert config.mag_weight == 0.5
#     assert config.max_chunk_size == 64


# @patch("src.tptt.modeling_tptt.AutoModelForCausalLM.from_pretrained")
# @patch("src.tptt.injection.inject_linear_attention")
# def test_tpttmodel_init(
#    mock_inject,
#   mock_model,
#    dummy_model,
#    cache,
#    dummy_tptt_config,
# ):
#   """Test TpttModel initialization with mocked dependencies."""
#    # Arrange: set up return values for the mocks
#    mock_model.return_value = dummy_model
#    mock_inject.return_value = (dummy_model, cache)
#
#    # Act: instantiate the TpttModel
#    tptt = TpttModel(dummy_tptt_config)
#
#    # Assert
#    assert tptt.config == dummy_tptt_config
#    assert tptt.backbone is dummy_model


def test_pipeline_initialization(dummy_pipeline_components):
    """Test pipeline initialization with valid components."""
    model, tokenizer = dummy_pipeline_components
    pipeline = TpttPipeline(model=model, tokenizer=tokenizer)
    assert pipeline.model is model
    assert pipeline.tokenizer is tokenizer


def test_sanitize_parameters(dummy_pipeline_components):
    """Test parameter sanitization logic."""
    model, tokenizer = dummy_pipeline_components
    pipeline = TpttPipeline(model, tokenizer)
    pre, forward, post = pipeline._sanitize_parameters(test_param=1)
    assert pre == {}
    assert forward == {}
    assert post == {}


def test_preprocess(dummy_pipeline_components):
    """Test input preprocessing with tokenization."""
    model, tokenizer = dummy_pipeline_components
    pipeline = TpttPipeline(model, tokenizer)
    test_prompt = "Test input"
    result = pipeline.preprocess(test_prompt)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)


def test_forward(dummy_pipeline_components):
    """Test model forward pass with generation."""
    model, tokenizer = dummy_pipeline_components
    pipeline = TpttPipeline(model, tokenizer, device="cpu")
    model_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    # Remplace la méthode generate par un mock
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4]]))
    output = pipeline._forward(model_inputs, max_new_tokens=50, do_sample=True)
    model.generate.assert_called_once()
    assert "generated_ids" in output


def test_postprocess(dummy_pipeline_components):
    """Test output postprocessing and decoding."""
    model, tokenizer = dummy_pipeline_components
    pipeline = TpttPipeline(model, tokenizer)
    mock_output = {"generated_ids": torch.tensor([[1, 2, 3, 4]])}
    result = pipeline.postprocess(mock_output)
    assert isinstance(result, list)
    assert "generated_text" in result[0]
    assert isinstance(result[0]["generated_text"], str)
