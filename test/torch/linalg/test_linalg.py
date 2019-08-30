import pytest
import torch as th
import syft as sy
from syft.frameworks.torch.linalg import inv_sym
from syft.frameworks.torch.linalg import qr


def test_inv_sym(hook, workers):
    """
    Testing inverse of symmetric matrix with MPC
    """
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")

    x = th.Tensor([[0.4627, 0.8224], [0.8224, 2.4084]])

    x = x.fix_precision(precision_fractional=6).share(bob, alice, crypto_provider=crypto_prov)
    gram = x.matmul(x.t())
    gram_inv = inv_sym(gram)

    gram_inv = gram_inv.get().float_precision()
    gram = gram.get().float_precision()

    diff = (gram_inv - gram.inverse()).abs()
    assert (diff < 1e-3).all()


def test_qr(hook, workers):
    """
    Testing QR decomposition with remote matrix
    """

    bob = workers["bob"]
    n_cols = 5
    n_rows = 10
    t = th.randn([n_rows, n_cols])
    Q, R = qr(t.send(bob), mode="complete")
    Q = Q.get()
    R = R.get()

    # Check if Q is orthogonal
    I = Q @ Q.t()
    assert ((th.eye(n_rows) - I).abs() < 1e-5).all()

    # Check if R is upper triangular matrix
    for col in range(n_cols):
        assert ((R[col + 1 :, col]).abs() < 1e-5).all()

    # Check if QR == t
    assert ((Q @ R - t).abs() < 1e-5).all()

    # test modes
    Q, R = qr(t.send(bob), mode="reduced")
    assert Q.shape == (n_rows, n_cols)
    assert R.shape == (n_cols, n_cols)

    Q, R = qr(t.send(bob), mode="complete")
    assert Q.shape == (n_rows, n_rows)
    assert R.shape == (n_rows, n_cols)

    R = qr(t.send(bob), mode="r")
    assert R.shape == (n_cols, n_cols)
