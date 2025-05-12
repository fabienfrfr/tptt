import torch.nn as nn

from liza.injection import CustomInjectConfig, get_custom_injected_model


def test_get_custom_injected_model(base_attn):
    model = nn.Sequential(base_attn)
    config = CustomInjectConfig(target_modules=["0"], operator="delta_rule")
    new_model = get_custom_injected_model(model, config)
    from liza.mpa import ParallelFLAAttention

    assert isinstance(new_model[0], ParallelFLAAttention)


def test_injection_module_not_found():
    model = nn.Sequential(nn.Linear(10, 10))
    config = CustomInjectConfig(target_modules=["42"], operator="delta_rule")
    new_model = get_custom_injected_model(model, config)
    assert isinstance(new_model[0], nn.Linear)
