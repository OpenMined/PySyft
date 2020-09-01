import pytest

import torch
from syft.frameworks.torch.mpc.falcon.falcon_helper import FalconHelper

import itertools


TEST_VALS = [(x, y, x ^ y) for x, y in itertools.product(torch.LongTensor([0, 1]), repeat=2)]


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_public_xor(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = x.share(bob, alice, james, protocol="falcon", field=2)
    assert (FalconHelper.xor(x, y).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_private_xor(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = x.share(bob, alice, james, protocol="falcon", field=2)
    y = y.share(bob, alice, james, protocol="falcon", field=2)
    assert (FalconHelper.xor(x, y).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("bit_select", [0, 1])
def test_select_share(bit_select, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]

    x = torch.tensor([0, 1, 2])
    x_shared = x.share(*workers, protocol="falcon")

    y = torch.tensor([-3, 0, 1])
    y_shared = y.share(*workers, protocol="falcon")

    b_shared = torch.tensor(bit_select).share(*workers, protocol="falcon", field=2)

    plaintext = FalconHelper.select_share(b_shared, x_shared, y_shared).reconstruct()

    if bit_select:
        assert (plaintext == y).all()
    else:
        assert (plaintext == x).all()


@pytest.mark.parametrize("beta", [0, 1])
def test_determine_sign(beta, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]

    x = torch.tensor([-4, 5, 6])
    x_shared = x.share(*workers, protocol="falcon")

    ring_size = x_shared.ring_size
    shape = x_shared.shape

    expected_plaintext = (-1) ** beta * x

    if beta:
        beta = torch.ones(size=shape, dtype=torch.long).share(
            *workers, protocol="falcon", field=ring_size
        )
    else:
        beta = torch.zeros(size=shape, dtype=torch.long).share(
            *workers, protocol="falcon", field=ring_size
        )

    plaintext = FalconHelper.determine_sign(x_shared, beta).reconstruct()

    assert (expected_plaintext == plaintext).all()
