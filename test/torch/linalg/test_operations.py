import pytest
import torch
import syft as sy
from syft.frameworks.torch.linalg import inv_sym
from syft.frameworks.torch.linalg import qr
from syft.frameworks.torch.linalg.operations import _norm_mpc
from test.efficiency_tests.assertions import assert_time


def test_inv_sym(hook, workers):
    """
    Testing inverse of symmetric matrix with MPC
    """
    torch.manual_seed(42)  # Truncation might not always work so we set the random seed

    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = workers["james"]

    x = torch.Tensor([[0.4627, 0.8224], [0.8224, 2.4084]])

    x = x.fix_precision(precision_fractional=6).share(bob, alice, crypto_provider=crypto_prov)
    gram = x.matmul(x.t())
    gram_inv = inv_sym(gram)

    gram_inv = gram_inv.get().float_precision()
    gram = gram.get().float_precision()

    diff = (gram_inv - gram.inverse()).abs()
    assert (diff < 1e-3).all()


@assert_time(max_time=20)
def test_norm_mpc(hook, workers):
    """
    Testing computation of vector norm on an AdditiveSharedTensor
    """
    torch.manual_seed(42)  # Truncation might not always work so we set the random seed
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = workers["james"]

    n = 100
    t = torch.randn([n])
    t_sh = t.fix_precision(precision_fractional=6).share(bob, alice, crypto_provider=crypto_prov)
    norm_sh = _norm_mpc(t_sh, norm_factor=n ** (1 / 2))
    norm = norm_sh.copy().get().float_precision()

    assert (norm - torch.norm(t)).abs() < 1e-4


def test_qr(hook, workers):
    """
    Testing QR decomposition with remote matrix
    """
    torch.manual_seed(42)  # Truncation might not always work so we set the random seed

    bob = workers["bob"]
    n_cols = 5
    n_rows = 10
    t = torch.randn([n_rows, n_cols])
    Q, R = qr(t.send(bob), mode="complete")
    Q = Q.get()
    R = R.get()

    # Check if Q is orthogonal
    I = Q @ Q.t()
    assert ((torch.eye(n_rows) - I).abs() < 1e-5).all()

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


def test_qr_mpc(hook, workers):
    """
    Testing QR decomposition with an AdditiveSharedTensor
    """
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = workers["james"]

    torch.manual_seed(0)  # Truncation might not always work so we set the random seed
    n_cols = 3
    n_rows = 3
    t = torch.randn([n_rows, n_cols])
    t_sh = t.fix_precision(precision_fractional=6).share(bob, alice, crypto_provider=crypto_prov)

    Q, R = qr(t_sh, norm_factor=3 ** (1 / 2), mode="complete")
    Q = Q.get().float_precision()
    R = R.get().float_precision()

    # Check if Q is orthogonal
    I = Q @ Q.t()
    assert ((torch.eye(n_rows) - I).abs() < 1e-2).all()

    # Check if R is upper triangular matrix
    for col in range(n_cols - 1):
        assert ((R[col + 1 :, col]).abs() < 1e-2).all()

    # Check if QR == t
    assert ((Q @ R - t).abs() < 1e-2).all()
