import torch

from liza.fla import delta_rule_closed_form, get_fla_operator, gla_recurrent


def test_gla_recurrent_shapes():
    B, H, N, D = 2, 3, 8, 4
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    g = torch.randn(B, H, N, 1)
    out, _ = gla_recurrent(q, k, v, g)
    assert out.shape == (B, H, N, D)

def test_delta_rule_closed_form_shapes():
    B, H, N, D = 2, 3, 8, 4
    v = torch.randn(B, H, N, D)
    g = torch.randn(B, H, N, D)
    out, _ = delta_rule_closed_form(None, None, v, g)
    assert out.shape == (B, H, N, D)
