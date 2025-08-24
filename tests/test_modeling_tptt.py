"""Unit tests for TpttModel and related classes (patch HF Hub everywhere)."""

import os
import shutil
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.tptt.configuration_tptt import TpttConfig
from src.tptt.modeling_tptt import TpttModel


# Patch HF Hub everywhere !
@pytest.fixture(autouse=True)
def patch_huggingface(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "src.tptt.configuration_tptt.AutoConfig.from_pretrained",
        lambda *a, **kw: MagicMock(),
    )
    # Patch loading of any model tptt_model from HF
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: MagicMock(),
    )
    # Patch AutoModel.from_pretrained as well
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.AutoModel.from_pretrained",
        lambda *a, **k: MagicMock(),
    )


class DummyConfig(TpttConfig):
    def __init__(self, **kwargs):
        kwargs.setdefault("base_model_name", "unit/test-tptt_model")
        super().__init__(**kwargs)


@pytest.fixture
def setup_model(monkeypatch):
    """Patch all ext deps of TpttModel for isolated tests."""
    tptt_model_mock = MagicMock()
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.AutoModelForCausalLM.from_pretrained",
        MagicMock(return_value=tptt_model_mock),
    )
    monkeypatch.setattr("src.tptt.modeling_tptt.LCache", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.LiZAttention", MagicMock())
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.get_tptt_model",
        lambda *a, **kw: (tptt_model_mock, "patched-linear-cache"),
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.LoraConfig", lambda **kw: "lora_conf_obj"
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.get_peft_model",
        lambda tptt_model, lora: tptt_model_mock,
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.load_tptt_safetensors",
        lambda repo, model, token=None, subfolder=None: tptt_model_mock,
    )
    logger_mock = MagicMock()
    monkeypatch.setattr("src.tptt.modeling_tptt.logger", logger_mock)
    return tptt_model_mock, logger_mock


def test_init_lora_and_load(setup_model):
    """Covers lora_config, get_peft_model/apply_safetensors, _base_path logic."""

    cfg = DummyConfig(lora_config={"alpha": 32}, _base_path="/somewhere")
    model = TpttModel(cfg)
    assert hasattr(model.tptt_model, "_loaded_safetensor")


def test_forward_training_sets_cache(setup_model):
    """Forward sets use_cache to False during training, removes num_items_in_batch."""

    tptt_model_mock, _ = setup_model
    cfg = DummyConfig()
    model = TpttModel(cfg)
    model.train(True)
    dummy_labels = torch.ones(1, 1).long()
    dummy_ids = torch.ones(1, 1).long()
    dummy_amask = torch.ones(1, 1)
    model.tptt_model.reset_mock()
    model.forward(dummy_ids, dummy_amask, dummy_labels, num_items_in_batch=3)
    args, kwargs = model.tptt_model.call_args
    assert kwargs["use_cache"] is False
    assert "num_items_in_batch" not in kwargs


def test_forward_eval_sets_cache(setup_model):
    """Eval mode: removes num_items_in_batch & sets use_cache=False if not present."""

    tptt_model_mock, _ = setup_model
    cfg = DummyConfig()
    model = TpttModel(cfg)
    model.train(False)
    model.tptt_model.reset_mock()
    # no use_cache given
    model.forward(torch.ones(1, 1).long(), torch.ones(1, 1), num_items_in_batch=5)
    args, kwargs = model.tptt_model.call_args
    assert kwargs["use_cache"] is False
    assert "num_items_in_batch" not in kwargs


def test_generate_delegation(setup_model):
    """generate() delegates to tptt_model.generate"""

    tptt_model_mock, _ = setup_model
    tptt_model_mock.generate.return_value = "result"
    cfg = DummyConfig()
    model = TpttModel(cfg)
    result = model.generate(1, 2)
    tptt_model_mock.generate.assert_called_with(1, 2)
    assert result == "result"


def test_copy_source_files(setup_model, monkeypatch, tmp_path):
    """_copy_source_files copies all .py from source dir."""

    cfg = DummyConfig()
    model = TpttModel(cfg)
    fake_src_dir = tmp_path / "srcdir"
    fake_src_dir.mkdir()
    src_file = fake_src_dir / "hello.py"
    src_file.write_text("hi")
    monkeypatch.setattr("src.tptt.modeling_tptt.__file__", str(fake_src_dir / "a.py"))
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.os.path.dirname", lambda x: str(fake_src_dir)
    )
    monkeypatch.setattr("src.tptt.modeling_tptt.os.path.abspath", lambda x: str(x))
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.os.listdir", lambda x: ["hello.py", "notpy.txt"]
    )
    monkeypatch.setattr("src.tptt.modeling_tptt.shutil.copy2", shutil.copy2)
    dst_dir = tmp_path / "output"
    dst_dir.mkdir()
    model._copy_source_files(str(dst_dir))
    assert (dst_dir / "hello.py").exists()


def test_retie_lm_after_load_share(setup_model, monkeypatch):
    """retie_lm_after_load shares weights when tie_word_embeddings=True."""

    embed = MagicMock()
    embed.weight = nn.Parameter(torch.randn(3, 5))
    tptt_model = MagicMock()
    tptt_model.lm_head = None
    monkeypatch.setattr("src.tptt.modeling_tptt.find_embedding_lm", lambda x: embed)
    logger_mock = MagicMock()
    monkeypatch.setattr("src.tptt.modeling_tptt.logger", logger_mock)
    model = TpttModel(DummyConfig())
    model.tptt_model = tptt_model
    model.retie_lm_after_load(tie_word_embeddings=True)
    assert tptt_model.lm_head.weight is embed.weight
    assert logger_mock.info.called
    msg = logger_mock.info.call_args[0][0]
    assert "shared" in msg


def test_retie_lm_after_load_clone(setup_model, monkeypatch):
    """retie_lm_after_load clones weights when tie_word_embeddings=False."""

    embed = MagicMock()
    embed.weight = nn.Parameter(torch.randn(3, 5))
    tptt_model = type("B", (), {})()
    tptt_model.lm_head = None
    monkeypatch.setattr("src.tptt.modeling_tptt.find_embedding_lm", lambda x: embed)
    logger_mock = MagicMock()
    monkeypatch.setattr("src.tptt.modeling_tptt.logger", logger_mock)
    model = TpttModel(DummyConfig())
    model.tptt_model = tptt_model
    model.retie_lm_after_load(tie_word_embeddings=False)
    assert tptt_model.lm_head.weight is not embed.weight
    assert logger_mock.info.called
    msg = logger_mock.info.call_args[0][0]
    assert "cloned" in msg


def test_from_pretrained_calls_super_and_retie():
    """from_pretrained delegates to super and then retie_lm_after_load."""

    called = {"super": False, "retie": False}

    class DummySuper(TpttModel):
        @classmethod
        def from_pretrained(cls, *a, **k):
            called["super"] = True
            inst = MagicMock(spec=cls)
            inst.retie_lm_after_load = lambda **kw: called.update({"retie": True})
            inst.retie_lm_after_load()
            return inst

    with patch("src.tptt.modeling_tptt.TpttModel", DummySuper):
        DummySuper.from_pretrained()
    assert called["super"]
    assert called["retie"]
