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
    # Patch loading of any config from HF
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "src.tptt.configuration_tptt.AutoConfig.from_pretrained",
        lambda *a, **kw: MagicMock(),
    )
    # Patch loading of any model backbone from HF
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: MagicMock(),
    )


class DummyConfig(TpttConfig):
    def __init__(self, **kwargs):
        kwargs.setdefault("base_model_name", "unit/test-backbone")
        super().__init__(**kwargs)


@pytest.fixture
def setup_model(monkeypatch):
    """Patch all ext deps of TpttModel for isolated tests."""
    backbone_mock = MagicMock()
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.AutoModelForCausalLM.from_pretrained",
        MagicMock(return_value=backbone_mock),
    )
    monkeypatch.setattr("src.tptt.modeling_tptt.LCache", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.LiZAttention", MagicMock())
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.get_tptt_model",
        lambda *a, **kw: (backbone_mock, "patched-linear-cache"),
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.LoraConfig", lambda **kw: "lora_conf_obj"
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.get_peft_model",
        lambda backbone, lora: backbone_mock,
    )
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.load_tptt_safetensors",
        lambda repo, model, token=None: backbone_mock,
    )
    logger_mock = MagicMock()
    monkeypatch.setattr("src.tptt.modeling_tptt.logger", logger_mock)
    return backbone_mock, logger_mock


def test_init_warn_force_attn(setup_model):
    """Warns if force_attn_implementation set in config."""

    _, logger_mock = setup_model
    cfg = DummyConfig(force_attn_implementation="cuda")
    TpttModel(cfg)
    assert logger_mock.warning.called
    msg = logger_mock.warning.call_args[0][0]
    assert "Attention implementation is:" in msg


def test_init_lora_and_load(setup_model):
    """Covers lora_config, get_peft_model/apply_safetensors, _base_path logic."""
    from src.tptt.modeling_tptt import TpttModel

    cfg = DummyConfig(lora_config={"alpha": 32}, _base_path="/somewhere")
    model = TpttModel(cfg)
    assert hasattr(model.backbone, "_loaded_safetensor")


def test_init_no_forced_attn(setup_model):
    """Backbone is loaded with no warning if force_attn_implementation is None."""

    _, logger_mock = setup_model
    logger_mock.warning.reset_mock()
    cfg = DummyConfig(force_attn_implementation=None)
    TpttModel(cfg)
    all_warnings = [str(call[0][0]) for call in logger_mock.warning.call_args_list]
    assert not any("Attention implementation is:" in log for log in all_warnings)


def test_inject_liza_attention(monkeypatch):
    """Delegates to get_tptt_model and returns as expected."""

    called = {}

    def fake_get_tptt_model(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return "bb", "lc"

    monkeypatch.setattr("src.tptt.modeling_tptt.get_tptt_model", fake_get_tptt_model)
    backbone = MagicMock()
    backbone.config = MagicMock()
    config = MagicMock()  # n'a pas besoin d'être DummyConfig pour test statique
    bb, lc = TpttModel.inject_liza_attention(backbone, config, "lc")
    assert bb == "bb" and lc == "lc"
    assert called["args"][0] == backbone


def test_forward_training_sets_cache(setup_model):
    """Forward sets use_cache to False during training, removes num_items_in_batch."""

    backbone_mock, _ = setup_model
    cfg = DummyConfig()
    model = TpttModel(cfg)
    model.train(True)
    dummy_labels = torch.ones(1, 1).long()
    dummy_ids = torch.ones(1, 1).long()
    dummy_amask = torch.ones(1, 1)
    model.backbone.reset_mock()
    model.forward(dummy_ids, dummy_amask, dummy_labels, num_items_in_batch=3)
    args, kwargs = model.backbone.call_args
    assert kwargs["use_cache"] is False
    assert "num_items_in_batch" not in kwargs


def test_forward_eval_sets_cache(setup_model):
    """Eval mode: removes num_items_in_batch & sets use_cache=False if not present."""

    backbone_mock, _ = setup_model
    cfg = DummyConfig()
    model = TpttModel(cfg)
    model.train(False)
    model.backbone.reset_mock()
    # no use_cache given
    model.forward(torch.ones(1, 1).long(), torch.ones(1, 1), num_items_in_batch=5)
    args, kwargs = model.backbone.call_args
    assert kwargs["use_cache"] is False
    assert "num_items_in_batch" not in kwargs


def test_generate_delegation(setup_model):
    """generate() delegates to backbone.generate"""

    backbone_mock, _ = setup_model
    backbone_mock.generate.return_value = "result"
    cfg = DummyConfig()
    model = TpttModel(cfg)
    result = model.generate(1, 2)
    backbone_mock.generate.assert_called_with(1, 2)
    assert result == "result"


def test_save_pretrained_calls_all(monkeypatch, setup_model, tmp_path):
    """Tests save_pretrained, _save_peft_weights, _copy_source_files."""

    cfg = DummyConfig()
    model = TpttModel(cfg)
    model._save_peft_weights = MagicMock()
    model._copy_source_files = MagicMock()
    super_save = MagicMock()
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.PreTrainedModel.save_pretrained", super_save
    )
    model.save_pretrained(str(tmp_path))
    assert super_save.called
    assert model._save_peft_weights.called
    assert model._copy_source_files.called


def test_save_peft_weights_removes_config(setup_model, tmp_path):
    """_save_peft_weights removes adapter_config.json if exists."""

    cfg = DummyConfig()
    model = TpttModel(cfg)
    model.backbone.save_pretrained = MagicMock()
    config_path = os.path.join(tmp_path, "adapter_config.json")
    with open(config_path, "w") as f:
        f.write("abc")
    assert os.path.exists(config_path)
    model._save_peft_weights(str(tmp_path))
    assert not os.path.exists(config_path)


def test_save_peft_weights_no_config(setup_model, tmp_path):
    """_save_peft_weights does nothing if adapter_config.json not there."""

    cfg = DummyConfig()
    model = TpttModel(cfg)
    model.backbone.save_pretrained = MagicMock()
    config_path = os.path.join(tmp_path, "adapter_config.json")
    if os.path.exists(config_path):
        os.remove(config_path)
    model._save_peft_weights(str(tmp_path))  # Doit juste passer


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
    backbone = MagicMock()
    backbone.lm_head = None
    monkeypatch.setattr("src.tptt.modeling_tptt.find_embedding_lm", lambda x: embed)
    logger_mock = MagicMock()
    monkeypatch.setattr("src.tptt.modeling_tptt.logger", logger_mock)
    model = TpttModel(DummyConfig())
    model.backbone = backbone
    model.retie_lm_after_load(tie_word_embeddings=True)
    assert backbone.lm_head.weight is embed.weight
    assert logger_mock.info.called
    msg = logger_mock.info.call_args[0][0]
    assert "shared" in msg


def test_retie_lm_after_load_clone(setup_model, monkeypatch):
    """retie_lm_after_load clones weights when tie_word_embeddings=False."""

    embed = MagicMock()
    embed.weight = nn.Parameter(torch.randn(3, 5))
    backbone = type("B", (), {})()
    backbone.lm_head = None
    monkeypatch.setattr("src.tptt.modeling_tptt.find_embedding_lm", lambda x: embed)
    logger_mock = MagicMock()
    monkeypatch.setattr("src.tptt.modeling_tptt.logger", logger_mock)
    model = TpttModel(DummyConfig())
    model.backbone = backbone
    model.retie_lm_after_load(tie_word_embeddings=False)
    assert backbone.lm_head.weight is not embed.weight
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
