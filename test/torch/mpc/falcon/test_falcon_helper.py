import torch
from syft.frameworks.torch.mpc.falcon.falcon_helper import FalconHelper


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


def test_select_share(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]
    x = torch.tensor([0, 1, 2]).share(*workers, protocol="falcon")
    y = torch.tensor([-3, 0, 1]).share(*workers, protocol="falcon")
    b_0 = torch.tensor(0).share(*workers, protocol="falcon", field=2)
    b_1 = torch.tensor(1).share(*workers, protocol="falcon", field=2)

    assert (FalconHelper.select_shares(b_0, x, y).reconstruct() == torch.tensor([0, 1, 2])).all()
    assert (FalconHelper.select_shares(b_1, x, y).reconstruct() == torch.tensor([-3, 0, 1])).all()


def test_determine_sign(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]
    x = torch.tensor([-4, 5, 6]).share(*workers, protocol="falcon")
    beta_0 = torch.tensor(0).share(*workers, protocol="falcon", field=2)
    beta_1 = torch.tensor(1).share(*workers, protocol="falcon", field=2)

    assert (FalconHelper.determine_sign(x, beta_0).reconstruct() == torch.tensor([-4, 5, 6])).all()
    assert (
        FalconHelper.determine_sign(x, beta_1).reconstruct() == (-1 * torch.tensor([-4, 5, 6]))
    ).all()
