import torch.nn as nn
from liza.injection import CustomInjectConfig, get_custom_injected_model

def test_get_custom_injected_model(dummy_base_attention):
    model = nn.Sequential(dummy_base_attention)
    config = CustomInjectConfig(target_modules=["0"], operator="gla")
    new_model = get_custom_injected_model(model, config)
    from liza.attention import ParallelFLAAttention
    assert isinstance(new_model[0], ParallelFLAAttention)
