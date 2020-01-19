import pytest
import torch
import syft as sy
from syft.frameworks.torch.linalg import inv_sym
from syft.frameworks.torch.linalg.operations import _norm_mpc
from test.efficiency_tests.assertions import assert_time


@assert_time(max_time=20)
def test_inv_sym(hook, workers):
    torch.manual_seed(42)  # Truncation might not always work so we set the random seed
    N = 100
    K = 2
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = workers["james"]

    x = torch.randn(N, K).fix_precision().share(bob, alice, crypto_provider=crypto_prov)
    gram = x.t().matmul(x)
    gram_inv = inv_sym(gram)
