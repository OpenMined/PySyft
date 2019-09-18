import pytest
import torch
import syft as sy
from test.efficiency_tests.assertions import assert_time


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
@assert_time(max_time=1)
def test_activation(activation, hook, workers):

    activation_func = torch.tanh if activation == "tanh" else torch.sigmoid

    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = workers["james"]

    x = torch.randn([10, 10]).fix_precision().share(bob, alice, crypto_provider=crypto_prov)
    activation_func(x)
