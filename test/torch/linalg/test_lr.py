import pytest
import torch
import syft as sy
from syft.frameworks.torch.linalg import BloomRegressor


def test_bloom(hook, workers):
    """
    Test BloomRegressor, i.e. distributed linear regression with MPC
    """
    bob = workers["bob"]
    alice = workers["alice"]
    jon = sy.VirtualWorker(hook, id="jon")
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")
    hbc_worker = sy.VirtualWorker(hook, id="hbc_worker")

    #### Simulate data ####

    K = 2  # number of features

    beta = torch.Tensor([1.0, 10.0]).view(-1, 1)  # "real" coefficients
    intercept = 3.0  # "real" intercept

    # Alice's data
    N1 = 10000
    X_alice = torch.randn(N1, K).send(alice)
    y_alice = X_alice @ beta.copy().send(alice) + intercept

    # Bob's data
    N2 = 20000
    X_bob = torch.randn(N2, K).send(bob)
    y_bob = X_bob @ beta.copy().send(bob) + intercept

    # Jon's data
    N3 = 15000
    X_jon = torch.randn(N3, K).send(jon)
    y_jon = X_jon @ beta.copy().send(jon) + intercept

    # Gather pointers into lists
    X_ptrs = [X_alice, X_bob, X_jon]
    y_ptrs = [y_alice, y_bob, y_jon]

    # Perform linear regression
    bloom_lr = BloomRegressor(crypto_prov, hbc_worker)
    bloom_lr.fit(X_ptrs, y_ptrs)

    assert abs(bloom_lr.intercept.item() - intercept) < 1e-5

    assert ((bloom_lr.coef - beta.squeeze()).abs() < 1e-4).all()

    # Test prediction
    # TODO
