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


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_xor_rearranges_operands(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = x.item()
    y = y.share(bob, alice, james, protocol="falcon", field=2)

    assert (FalconHelper.xor(x, y).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_xor_no_rearranges_operands(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = x.share(bob, alice, james, protocol="falcon", field=2)
    y = y.item()

    assert (FalconHelper.xor(x, y).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_xor_rearranges_operands_outside_range(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = x.item() - 2
    y = y.share(bob, alice, james, protocol="falcon", field=2)

    with pytest.raises(ValueError) as e:
        FalconHelper.xor(x, y)

    assert str(e.value) == "The integer value should be in {0, 1}"


def test_xor_invalid_ring_size(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    x = torch.tensor([0]).share(bob, alice, james, protocol="falcon", field=3)
    y = torch.tensor([1]).share(bob, alice, james, protocol="falcon", field=2)

    with pytest.raises(ValueError) as e:
        FalconHelper.xor(x, y)

    assert str(e.value) == "If both arguments are RST then they should be in ring size 2"

    x = torch.tensor([0]).share(bob, alice, james, protocol="falcon", field=2)
    y = torch.tensor([1]).share(bob, alice, james, protocol="falcon", field=3)

    with pytest.raises(ValueError) as e:
        FalconHelper.xor(x, y)

    assert str(e.value) == "If both arguments are RST then they should be in ring size 2"

    x = torch.tensor([0]).share(bob, alice, james, protocol="falcon", field=3)
    y = torch.tensor([1]).share(bob, alice, james, protocol="falcon", field=3)

    with pytest.raises(ValueError) as e:
        FalconHelper.xor(x, y)

    assert str(e.value) == "If both arguments are RST then they should be in ring size 2"


def test_xor_no_rst(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    x = 1
    y = 0

    with pytest.raises(ValueError) as e:
        FalconHelper.xor(x, y)

    assert str(e.value) == "One of the arguments should be a RST"


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
def test_evaluate(beta, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]

    x = torch.tensor([-4, 5, 6])
    x_shared = x.share(*workers, protocol="falcon")

    ring_size = x_shared.ring_size
    shape = x_shared.shape

    expected_plaintext = (-1) ** beta * x

    if beta:
        beta = torch.ones(size=shape).share(*workers, protocol="falcon", field=ring_size)
    else:
        beta = torch.zeros(size=shape).share(*workers, protocol="falcon", field=ring_size)

    plaintext = FalconHelper.negate_cond(x_shared, beta).reconstruct()

    assert (expected_plaintext == plaintext).all()
