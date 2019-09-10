import pytest
import torch
import syft as sy
from syft.frameworks.torch.linalg import inv_sym


def test_inv_sym(hook, workers):
    """
    Testing inverse of symmetric matrix with MPC
    """
    torch.manual_seed(42)  # Truncation might not always work so we set the random seed

    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")

    x = torch.Tensor([[0.4627, 0.8224], [0.8224, 2.4084]])

    x = x.fix_precision(precision_fractional=6).share(bob, alice, crypto_provider=crypto_prov)
    gram = x.matmul(x.t())
    gram_inv = inv_sym(gram)

    gram_inv = gram_inv.get().float_precision()
    gram = gram.get().float_precision()

    diff = (gram_inv - gram.inverse()).abs()
    assert (diff < 1e-3).all()
