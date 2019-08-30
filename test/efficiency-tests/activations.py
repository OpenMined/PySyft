import pytest
import time

import syft as sy
import torch as th


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_activation(activation, hook, workers):
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")

    if activation == "tanh":
        func = th.tanh
    else:
        func = th.sigmoid

    x = th.randn([10, 10]).fix_precision().share(bob, alice, crypto_provider=crypto_prov)
    t0 = time.time()
    func(x)
    dt = time.time() - t0

    # We expect the operation takes less than 1 second
    assert dt < 1
