# pylint: disable=protected-access
"""Unit tests for the TpttModel and related classes."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import DynamicCache

from src.tptt.modeling_tptt import LiZAttention, get_tptt_model

MODULE = "src.tptt.modeling_tptt"  # for patch


# --- Fixtures and dummies ---


class DummyBaseConfig:
    def __init__(self, hidden_size=48):
        self.hidden_size = hidden_size
        self.num_heads = 4
        self.num_key_value_heads = 4


class DummyAttn:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyConfig:
    def __init__(self, **kwargs):
        # always set hidden_size
        self.hidden_size = kwargs.get("hidden_size", 16)
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def patch_split_qkv(monkeypatch):
    """Patch split_qkv globally (simulate QKV split like in OpenELM/GPTNeo case)"""
    monkeypatch.setattr(
        "src.tptt.modeling_tptt.split_qkv",
        lambda base_attn, qkv: qkv.chunk(3, dim=-1),
    )


def dummy_tensor():
    """Create a dummy tensor compatible with all projection methods (batch, seq, dim)"""
    return torch.randn(2, 4, 12)


def make_liza_with_base_attn(base_attn):
    """Helper to create a LiZAttention object with only required config."""
    mock_config = MagicMock()
    # Fake attributes for attention params
    mock_config.hidden_size = 12 * 4  # 4 heads * 12 dim
    mock_config.num_heads = 4
    mock_config.num_key_value_heads = 4
    return LiZAttention(base_attn=base_attn, layer_idx=0, base_config=mock_config)


def test_proj_c_attn_and_c_proj():
    """Test GPT-2 style branch: c_attn and c_proj attributes."""
    t = torch.randn(2, 4, 36)  # will chunk into 3 tensors of shape (2,4,12)
    base_attn = MagicMock()
    base_attn.c_attn = MagicMock(return_value=t)
    base_attn.c_proj = MagicMock()
    # Remove interfering attributes
    for n in ["q_proj", "qkv_proj", "query", "key", "value", "dense"]:
        if hasattr(base_attn, n):
            delattr(base_attn, n)
    liz = make_liza_with_base_attn(base_attn)
    q, k, v, out_proj = liz._apply_shared_projections(torch.randn(2, 4, 12))
    base_attn.c_attn.assert_called_once()
    assert out_proj == base_attn.c_proj
    assert q.shape == k.shape == v.shape


def test_proj_query_key_value_with_dense():
    """Test BERT/ViT style: query, key, value and dense (output projection)."""
    t = dummy_tensor()
    base_attn = MagicMock()
    base_attn.query = MagicMock(return_value=t)
    base_attn.key = MagicMock(return_value=t)
    base_attn.value = MagicMock(return_value=t)
    base_attn.dense = MagicMock()  # output
    # Remove interfering attributes
    for n in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "c_attn", "c_proj"]:
        if hasattr(base_attn, n):
            delattr(base_attn, n)
    liz = make_liza_with_base_attn(base_attn)
    q, k, v, out_proj = liz._apply_shared_projections(torch.randn(2, 4, 12))
    base_attn.query.assert_called_once()
    base_attn.key.assert_called_once()
    base_attn.value.assert_called_once()
    assert out_proj == base_attn.dense
    assert q.shape == k.shape == v.shape


def test_proj_query_key_value_without_dense():
    """Test BERT/ViT style: query, key, value present, but no dense projection."""
    t = dummy_tensor()
    base_attn = MagicMock()
    base_attn.query = MagicMock(return_value=t)
    base_attn.key = MagicMock(return_value=t)
    base_attn.value = MagicMock(return_value=t)
    # Do NOT set .dense output
    for n in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "c_attn",
        "c_proj",
        "dense",
    ]:
        if hasattr(base_attn, n):
            delattr(base_attn, n)
    liz = make_liza_with_base_attn(base_attn)
    q, k, v, out_proj = liz._apply_shared_projections(torch.randn(2, 4, 12))
    base_attn.query.assert_called_once()
    base_attn.key.assert_called_once()
    base_attn.value.assert_called_once()
    assert out_proj is None
    assert q.shape == k.shape == v.shape


def test_proj_raises_if_no_supported_proj():
    """Test ValueError when no valid projection attributes found."""
    base_attn = MagicMock()
    # Remove all known projection attributes
    for n in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "c_attn",
        "c_proj",
        "query",
        "key",
        "value",
        "dense",
    ]:
        if hasattr(base_attn, n):
            delattr(base_attn, n)
    liz = make_liza_with_base_attn(base_attn)
    with pytest.raises(ValueError, match="Unsupported attention module"):
        liz._apply_shared_projections(torch.randn(2, 4, 12))


def test_apply_shared_projections_q_proj():
    """Test branch with q_proj, k_proj, v_proj, o_proj."""
    t = dummy_tensor()
    base_attn = MagicMock()
    base_attn.q_proj.return_value = t
    base_attn.k_proj.return_value = t
    base_attn.v_proj.return_value = t
    base_attn.o_proj = MagicMock()

    liz = make_liza_with_base_attn(base_attn)
    _, _, _, out_proj = liz._apply_shared_projections(t)
    base_attn.q_proj.assert_called_once_with(t)
    base_attn.k_proj.assert_called_once_with(t)
    base_attn.v_proj.assert_called_once_with(t)
    assert out_proj == base_attn.o_proj


def test_apply_shared_projections_qkv_proj():
    """Test only the qkv_proj branch."""
    t = torch.randn(2, 4, 36)
    base_attn = MagicMock()
    base_attn.qkv_proj = MagicMock(return_value=t)
    base_attn.out_proj = MagicMock()
    for n in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "c_attn",
        "c_proj",
        "query",
        "key",
        "value",
        "dense",
    ]:
        if hasattr(base_attn, n):
            delattr(base_attn, n)
    liz = make_liza_with_base_attn(base_attn)
    _, _, _, out_proj = liz._apply_shared_projections(t)
    base_attn.qkv_proj.assert_called_once_with(t)
    assert out_proj == base_attn.out_proj


# --- _get_attention_parameters ---


@pytest.mark.parametrize(
    "attn_kwargs, config_kwargs, expected",
    [
        # 1. All params present on base_attn
        (
            {
                "num_heads": 2,
                "head_dim": 8,
                "num_kv_heads": 2,
                "num_key_value_groups": 2,
            },
            {},
            (2, 8, 2, 2),
        ),
        # 2. missing some attrs, fallback head_dim = hidden_size // num_heads
        ({"num_heads": 2, "num_kv_heads": 2}, {}, (2, 8, 2, 1)),
        # 3. Everything via config, pristine
        ({}, {"num_heads": 3, "head_dim": 5, "num_key_value_heads": 7}, (3, 5, 7, 0)),
        # 4. only hidden_size and num_heads, test fallback for head_dim
        ({}, {"hidden_size": 16, "num_heads": 2}, (2, 8, 2, 1)),
    ],
)
def test_get_attention_parameters(attn_kwargs, config_kwargs, expected):
    import src.tptt.modeling_tptt as modeling

    modeling.LinearAttention = MagicMock()
    modeling.CausalAvgPool1d = MagicMock()
    # Always ensure a hidden_size is present!
    if "hidden_size" not in config_kwargs:
        config_kwargs["hidden_size"] = 16
    base_attn = DummyAttn(**attn_kwargs)
    base_config = DummyConfig(**config_kwargs)
    liz = LiZAttention(base_attn, 0, base_config)
    params = liz._get_attention_parameters(base_attn, base_config)
    assert params == expected


def test_lizattention_init_with_recurrent_config(monkeypatch):
    """Test __init__ with explicit recurrent_config."""

    received_kwargs = {}

    def fake_linear_attention(*args, **kwargs):
        received_kwargs.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", fake_linear_attention)
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())
    base_attn = MagicMock()
    base_config = DummyConfig(hidden_size=32, num_heads=2, num_key_value_heads=2)
    r_config = {"order": 2, "gate_type": "k", "linear": True, "trick": "rotative"}
    LiZAttention(base_attn, 0, base_config, recurrent_config=r_config)
    assert received_kwargs["recurrent_config"] == r_config


def test_apply_mag_classic(monkeypatch):
    """Test classic (no cross_gate) MAG logic and left-padding."""

    class DummyLiz:
        mag_weight = 0.6
        cross_gate = False

    liz = DummyLiz()
    # simulate padding, last dim as features
    liz._apply_mag = LiZAttention._apply_mag.__get__(liz)
    o_lin = torch.ones(2, 5, 3)
    o_base = torch.ones(2, 4, 3)
    out = liz._apply_mag(o_lin, o_base)
    assert out.shape == (2, 4, 3)
    assert torch.allclose(out, torch.ones_like(out))


# --- __init__ linear_precision, recurrent_config default, assign attrs ---


def test_lizattention_init_precision_string_and_dtype(monkeypatch):
    """Test init with linear_precision string and dtype."""
    # patch LinearAttention/CausalAvgPool1d to avoid errors
    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())
    base_attn = MagicMock()
    base_config = DummyBaseConfig()
    # test string version
    liz1 = LiZAttention(base_attn, 0, base_config, linear_precision="float16")
    assert liz1.linear_precision == torch.float16
    # test dtype version
    liz2 = LiZAttention(base_attn, 0, base_config, linear_precision=torch.float32)
    assert liz2.linear_precision == torch.float32


def test_lizattention_scaling_and_attrs(monkeypatch):
    """Test scaling and all attributes set in __init__."""
    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())
    base_attn = MagicMock(
        num_heads=2, head_dim=4, num_kv_heads=2, num_key_value_groups=2
    )
    base_config = DummyBaseConfig(hidden_size=8)
    liz = LiZAttention(base_attn, 0, base_config)
    assert hasattr(liz, "num_heads")
    assert hasattr(liz, "head_dim")
    assert hasattr(liz, "scaling")
    assert hasattr(liz, "linear_attn")
    assert abs(liz.scaling - 0.5) < 1e-4


# --- _process_self_attn ---


def make_liza_for_process(monkeypatch, max_self_attn_length=None):
    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())
    liz = LiZAttention(
        MagicMock(), 0, DummyBaseConfig(), max_self_attn_length=max_self_attn_length
    )
    return liz


def test_process_self_attn_basic(monkeypatch):
    """Test _process_self_attn base logic."""
    liz = make_liza_for_process(monkeypatch)
    base_attn = liz.base_attn
    # Only return one item
    base_attn.return_value = torch.randn(2, 4, 12)
    hs = torch.randn(2, 4, 12)
    o_base, attn_weights, present_key_value, mode = liz._process_self_attn(hs, None, {})
    assert isinstance(o_base, torch.Tensor)
    assert attn_weights is None
    assert present_key_value is None
    assert mode == 1


def test_process_self_attn_tuple2(monkeypatch):
    """Test tuple of len 2 from base_attn."""
    liz = make_liza_for_process(monkeypatch)
    liz.base_attn.return_value = (torch.randn(2, 4, 12), torch.randn(2, 4, 4))
    hs = torch.randn(2, 4, 12)
    o_base, _, _, mode = liz._process_self_attn(hs, None, {})
    assert o_base.shape == (2, 4, 12)
    assert mode == 2


def test_process_self_attn_tuple3(monkeypatch):
    """Test tuple of len 3 from base_attn."""
    liz = make_liza_for_process(monkeypatch)
    liz.base_attn.return_value = (torch.randn(2, 4, 12), torch.randn(2, 4, 4), "pkv")
    hs = torch.randn(2, 4, 12)
    o_base, _, present_key_value, mode = liz._process_self_attn(hs, None, {})
    assert o_base.shape == (2, 4, 12)
    assert mode == 3
    assert present_key_value == "pkv"


def test_process_self_attn_truncate(monkeypatch):
    """Test max_self_attn_length truncation path, with position_embeddings."""
    liz = make_liza_for_process(monkeypatch, max_self_attn_length=2)
    base_attn = liz.base_attn
    base_attn.return_value = torch.randn(2, 2, 12)
    # Patch truncate_attention_mask
    with patch(
        "src.tptt.modeling_tptt.truncate_attention_mask",
        return_value=(torch.ones(2, 2, 12), torch.ones(2, 2)),
    ):
        hs = torch.randn(2, 4, 12)
        mask = torch.ones(2, 4)
        args = {
            "position_embeddings": (torch.ones(2, 5), torch.ones(2, 5)),
            "past_key_value": MagicMock(),
        }
        liz.layer_idx = 0
        args["past_key_value"].__len__.return_value = 1
        liz._process_self_attn(hs, mask, args)
        # The important thing: test does not crash and calls crop


def test_process_self_attn_unexpected_length(monkeypatch):
    """Test ValueError if base_attn returns a tuple of another length."""
    liz = make_liza_for_process(monkeypatch)
    liz.base_attn.return_value = (torch.randn(2, 4, 12),)
    hs = torch.randn(2, 4, 12)
    with pytest.raises(ValueError):
        liz._process_self_attn(hs, None, {})


# --- _prepare_attn_mixin ---


def test_prepare_attn_mixin_base_scale(monkeypatch):
    """Test scaling in _prepare_attn_mixin."""
    liz = make_liza_for_process(monkeypatch)
    liz.base_scale_attn = True
    o_lin = torch.ones(2, 4, 12)
    o_base = torch.ones(2, 4, 12)
    out_lin, out_base = liz._prepare_attn_mixin(o_lin, o_base, torch.float32)
    assert torch.allclose(out_lin, out_base)


# --- _apply_mag ---


def test_apply_mag_cross_gate(monkeypatch):
    """Test cross_gate True logic."""
    liz = make_liza_for_process(monkeypatch)
    liz.mag_weight = 0.5
    liz.cross_gate = True
    o_lin = torch.ones(2, 4, 3)
    o_base = torch.zeros(2, 4, 3)
    out = liz._apply_mag(o_lin, o_base)
    # Result: out = o_lin*mag + o_base*(1-mag) + same*cross = o_lin*0.5 + cross_terms
    assert torch.any(out != 0)


# --- forward ---


def test_forward_main_logic(monkeypatch):
    """Test forward main flow, return shape and use_cache logic."""
    # Patch submodules
    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())
    # Patch ensure_stability everywhere for deterministic output
    monkeypatch.setattr("src.tptt.modeling_tptt.ensure_stability", lambda x, **kw: x)
    # Patch _apply_shared_projections, _process_self_attn, _prepare_attn_mixin, _apply_mag
    liz = LiZAttention(MagicMock(), 0, DummyBaseConfig())
    monkeypatch.setattr(
        liz,
        "_apply_shared_projections",
        lambda x: (
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            None,
        ),
    )
    monkeypatch.setattr(
        liz, "linear_attn", MagicMock(return_value=torch.zeros(2, 2, 2))
    )
    monkeypatch.setattr(
        liz, "_process_self_attn", lambda x, y, z: (torch.zeros(2, 2, 2), None, None, 1)
    )
    monkeypatch.setattr(liz, "_prepare_attn_mixin", lambda x, y, z, eps: (x, y))
    monkeypatch.setattr(liz, "_apply_mag", lambda x, y: torch.zeros(2, 2, 2))
    hs = torch.zeros(2, 2, 2)
    result = liz.forward(hs)
    assert result.shape == (2, 2, 2)


def test_forward_expected_attn_modes(monkeypatch):
    """Test forward returns correct number of values depending on expected_attn_mode."""
    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())
    # Patch _apply_shared_projections etc
    liz = LiZAttention(MagicMock(), 0, DummyBaseConfig())
    monkeypatch.setattr(
        liz,
        "_apply_shared_projections",
        lambda x: (
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            None,
        ),
    )
    monkeypatch.setattr(
        liz, "linear_attn", MagicMock(return_value=torch.zeros(2, 2, 2))
    )
    monkeypatch.setattr(
        liz,
        "_process_self_attn",
        lambda x, y, z: (
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            3,
        ),
    )
    monkeypatch.setattr(liz, "_prepare_attn_mixin", lambda x, y, z, eps: (x, y))
    monkeypatch.setattr(liz, "_apply_mag", lambda x, y: torch.zeros(2, 2, 2))
    hs = torch.zeros(2, 2, 2)
    out = liz.forward(hs)
    # When mode==3, return tuple of 3
    assert isinstance(out, tuple) and len(out) == 3


def test_process_self_attn_calls_crop(monkeypatch):
    """Test that past_key_value.crop(...) is called when expected (DynamicCache)."""

    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())

    class DummyDynamicCache(DynamicCache):
        """Dummy Cache"""

        def __init__(self):
            super().__init__()
            self.cropped = None

        def __len__(self):
            """Dummy len"""
            return 2

        def crop(self, value):
            """Dummy crop"""
            self.cropped = value

    cache = DummyDynamicCache()

    liz = LiZAttention(
        MagicMock(), 0, MagicMock(hidden_size=16), max_self_attn_length=5
    )

    hs = torch.randn(2, 4, 12)
    mask = torch.ones(2, 4)
    args = {
        "past_key_value": cache,
        "position_embeddings": (torch.ones(2, 8), torch.ones(2, 8)),
    }

    with patch(
        "src.tptt.modeling_tptt.truncate_attention_mask", return_value=(hs, mask)
    ):
        liz._process_self_attn(hs, mask, args)

    # crop must have been called with 4 (5-1)
    assert cache.cropped == 4


def test_apply_mag_logs_when_close(monkeypatch, caplog):
    """Test that logger.info triggers if output_attention and softmax_weighted are close."""

    liz = LiZAttention(MagicMock(), 0, MagicMock(hidden_size=16))
    liz.mag_weight = 0.0  # so output = softmax_weighted
    liz.cross_gate = False
    o_lin = torch.zeros(2, 2, 2)
    o_base = torch.ones(2, 2, 2)
    with caplog.at_level("INFO"):
        liz._apply_mag(o_lin, o_base)
    # Check that the log message was emitted
    assert "[LOG] layer" in caplog.text


def test_forward_elif_use_cache_path(monkeypatch):
    """Test forward branch for elif 'use_cache' not in kwargs."""
    # Patch modules and stable functions
    monkeypatch.setattr("src.tptt.modeling_tptt.LinearAttention", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.CausalAvgPool1d", MagicMock())
    monkeypatch.setattr("src.tptt.modeling_tptt.ensure_stability", lambda x, **kw: x)
    liz = LiZAttention(MagicMock(), 0, MagicMock(hidden_size=16))
    # Patch all subfunctions to simply pass through
    monkeypatch.setattr(
        liz,
        "_apply_shared_projections",
        lambda x: (
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            None,
        ),
    )
    liz.linear_attn = MagicMock(return_value=torch.zeros(2, 2, 2))
    monkeypatch.setattr(
        liz, "_process_self_attn", lambda x, y, z: (torch.zeros(2, 2, 2), None, None, 1)
    )
    monkeypatch.setattr(liz, "_prepare_attn_mixin", lambda x, y, z, eps: (x, y))
    monkeypatch.setattr(liz, "_apply_mag", lambda x, y: torch.zeros(2, 2, 2))
    liz.train(False)  # Switch to eval
    hs = torch.zeros(2, 2, 2)
    # Call with a kwarg "past_key_value" but NO "use_cache"
    result = liz.forward(hs, past_key_value="xxx")
    # Should not crash, output as usual
    assert torch.is_tensor(result)


def make_dummy_model(*module_names):
    """Create a dummy nn.Module with named submodules as attribute."""

    class DummyModel(torch.nn.Module):
        """Dummy model"""

        def __init__(self):
            super().__init__()
            for name in module_names:
                setattr(self, name, MagicMock())

        def named_modules(self):
            """Dummy named"""
            yield "", self
            for name in module_names:
                yield name, getattr(self, name)

    return DummyModel()


def make_dummy_model(*module_names):
    """Create a dummy nn.Module with named submodules as attributes."""

    class DummySubModule(torch.nn.Module):
        """Dummy sub module"""

        def __init__(self):
            super().__init__()

    class DummyModel(torch.nn.Module):
        """Dummy mode"""

        def __init__(self):
            super().__init__()
            for name in module_names:
                setattr(self, name, DummySubModule())

        def named_modules(self):
            """Dummy named"""
            yield "", self
            for name in module_names:
                yield name, getattr(self, name)

    return DummyModel()


def test_get_tptt_model_raise_on_no_target_module(monkeypatch):
    """It raises if no matching target module is found."""

    class DummyModel(torch.nn.Module):
        """Dummyy"""

        def named_modules(self):
            yield "", self

    monkeypatch.setattr("src.tptt.modeling_tptt.LCache", MagicMock())
    base_config = MagicMock()
    model = DummyModel()
    with pytest.raises(ValueError, match="Target modules"):
        get_tptt_model(model, base_config, target_modules_names=["doesntexist"])
