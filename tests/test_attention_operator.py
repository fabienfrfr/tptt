import pytest
import torch

from liza.attention_operator import AttentionOperator


@pytest.fixture
def tensor_dim():
    return 64


@pytest.fixture
def chunk_size():
    return 32


@pytest.fixture
def seq_len():
    return 128


@pytest.fixture
def random_tensors(seq_len, tensor_dim):
    Q = torch.randn(seq_len, tensor_dim)
    K = torch.randn(seq_len, tensor_dim)
    V = torch.randn(seq_len, tensor_dim)
    beta = torch.randn(seq_len, tensor_dim)
    return Q, K, V, beta


@pytest.fixture
def operator(tensor_dim):
    return AttentionOperator(mode="delta_rule", head_dim=tensor_dim)


def test_forward_shape(operator, random_tensors, chunk_size, seq_len, tensor_dim):
    Q, K, V, beta = random_tensors
    O, _ = operator(Q, K, V, beta, chunk_size)
    assert O.shape == (seq_len, tensor_dim)


def test_chunk_size_1(operator, seq_len, tensor_dim):
    Q = K = V = beta = torch.ones(seq_len, tensor_dim)
    O, _ = operator(Q, K, V, beta, chunk_size=1)
    assert O.shape == (seq_len, tensor_dim)
