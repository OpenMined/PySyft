from .assertions import assert_time

import pytest

import syft as sy
import torch as th


@pytest.mark.parametrize("activation", [th.tanh, th.sigmoid])
@assert_time(max_time=1)
def test_activation(activation, hook, workers):
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")

    x = th.randn([10, 10]).fix_precision().share(bob, alice, crypto_provider=crypto_prov)
    activation(x)
