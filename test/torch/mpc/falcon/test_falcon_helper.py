import torch
from syft.frameworks.torch.mpc.falcon.faclon_helper import FalconHelper


def test_public_xor(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([0]).share(bob, alice, james, protocol="falcon", field=2)
    y = torch.tensor([1])
    a = torch.tensor([1]).share(bob, alice, james, protocol="falcon", field=2)
    b = torch.tensor([0])
    assert (FalconHelper.xor(x, y).reconstruct() == torch.tensor(1)).all()
    assert (FalconHelper.xor(x, b).reconstruct() == torch.tensor(0)).all()
    assert (FalconHelper.xor(a, y).reconstruct() == torch.tensor(0)).all()
    assert (FalconHelper.xor(a, b).reconstruct() == torch.tensor(1)).all()


def test_private_xor(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([0]).share(bob, alice, james, protocol="falcon", field=2)
    y = torch.tensor([1]).share(bob, alice, james, protocol="falcon", field=2)
    assert (FalconHelper.xor(x, y).reconstruct() == torch.tensor(1)).all()
    assert (FalconHelper.xor(x, x).reconstruct() == torch.tensor(0)).all()
    assert (FalconHelper.xor(y, y).reconstruct() == torch.tensor(0)).all()
    assert (FalconHelper.xor(y, x).reconstruct() == torch.tensor(1)).all()
