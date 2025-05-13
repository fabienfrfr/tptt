import torch

from liza.utils import get_valid_chunk_size, match_dim, repeat_kv


def test_repeat_kv():
    x = torch.randn(2, 4, 8, 64)
    repeated = repeat_kv(x, 2)
    assert repeated.shape == (2, 8, 8, 64)


def test_match_dim_expand():
    x = torch.randn(1, 10, 32)
    matched = match_dim(x, 1, 20)
    assert matched.shape == (1, 20, 32)


def test_match_dim_reduce():
    x = torch.randn(1, 10, 32)
    matched = match_dim(x, 1, 5)
    assert matched.shape == (1, 5, 32)


def test_get_valid_chunk_size():
    assert get_valid_chunk_size(127, 64) == 1
    assert get_valid_chunk_size(128, 64) == 64
    assert get_valid_chunk_size(120, 64) == 60
