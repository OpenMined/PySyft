import pytest
import torch
import syft as sy
from syft.frameworks.torch.linalg import BloomRegressor


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_bloom(fit_intercept, hook, workers):
    """
    Test BloomRegressor, i.e. distributed linear regression with MPC
    """

    bob = workers["bob"]
    alice = workers["alice"]
    james = workers["james"]
    crypto_prov = workers["james"]
    hbc_worker = workers["charlie"]

    ###### Simulate data ######

    K = 2  # number of features

    beta = torch.Tensor([1.0, 10.0]).view(-1, 1)  # "real" coefficients
    intercept = 3.0 if fit_intercept else 0  # "real" intercept

    # Alice's data
    torch.manual_seed(0)  # Truncation might not always work so we set the random seed
    N1 = 10000
    X_alice = torch.randn(N1, K).send(alice)
    y_alice = X_alice @ beta.copy().send(alice) + intercept

    # Bob's data
    torch.manual_seed(42)  # Setting another seed to avoid creation of singular matrices
    N2 = 20000
    X_bob = torch.randn(N2, K).send(bob)
    y_bob = X_bob @ beta.copy().send(bob) + intercept

    # James's data
    N3 = 15000
    X_james = torch.randn(N3, K).send(james)
    y_james = X_james @ beta.copy().send(james) + intercept

    # Gather pointers into lists
    X_ptrs = [X_alice, X_bob, X_james]
    y_ptrs = [y_alice, y_bob, y_james]

    # Perform linear regression
    bloom_lr = BloomRegressor(crypto_prov, hbc_worker, fit_intercept=fit_intercept)
    bloom_lr.fit(X_ptrs, y_ptrs)

    if fit_intercept:
        assert abs(bloom_lr.intercept.item() - intercept) < 1e-3

    assert ((bloom_lr.coef - beta.squeeze()).abs() < 1e-3).all()

    ###### Test prediction #######
    # Pointer tensor
    diff = bloom_lr.predict(X_alice) - y_alice.squeeze()
    assert (diff.get().abs() < 1e-3).all()

    # Local tensor
    X_local = X_alice.get()
    y_local = y_alice.get()
    diff = bloom_lr.predict(X_local) - y_local.squeeze()
    assert (diff.abs() < 1e-3).all()

    ##### Test summarize ######

    bloom_lr.summarize()
