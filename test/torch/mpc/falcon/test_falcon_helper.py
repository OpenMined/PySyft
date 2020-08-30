import pytest

import torch
from syft.frameworks.torch.mpc.falcon.falcon_helper import FalconHelper

TEST_VALS = list((x, y, x ^ y) for x, y in itertools.product(torch.LongTensor([0, 1]), repeat=2))


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_public_xor(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x_share = x.share(bob, alice, james, protocol="falcon", field=2)
    assert (FalconHelper.xor(x_share, y).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_private_xor(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x_share = x.share(bob, alice, james, protocol="falcon", field=2)
    y_share = y.share(bob, alice, james, protocol="falcon", field=2)
    assert (FalconHelper.xor(x_share, y_share).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_xor_rearranges_operands(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = x.item()
    y_share = y.share(bob, alice, james, protocol="falcon", field=2)

    assert (FalconHelper.xor(x, y_share).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_xor_no_rearranges_operands(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x_share = x.share(bob, alice, james, protocol="falcon", field=2)
    y = y.item()

    assert (FalconHelper.xor(x_share, y).reconstruct() == x_xor_y).all()


@pytest.mark.parametrize("x, y, x_xor_y", TEST_VALS)
def test_xor_rearranges_operands_outside_range(x, y, x_xor_y, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = x.item() - 2
    y_share = y.share(bob, alice, james, protocol="falcon", field=2)

    with pytest.raises(ValueError) as e:
        FalconHelper.xor(x, y_share)

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
