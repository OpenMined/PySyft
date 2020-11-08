import torch

from syft.frameworks.torch.mpc.falcon import falcon
from syft.frameworks.torch.mpc.falcon.falcon_helper import FalconHelper


def test_conv2d_public(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([[[[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4], [4, 5, 6, 7]]]]).share(
        bob, alice, james, protocol="falcon"
    )
    y = torch.tensor([[[[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4], [4, 5, 6, 7]]]])
    assert (
        falcon.conv2d(x, y, padding=1).reconstruct()
        == torch.tensor([[[[123, 180, 132], [224, 312, 224], [132, 180, 123]]]])
    ).all()


def test_conv2d_private(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([[[[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4], [4, 5, 6, 7]]]]).share(
        bob, alice, james, protocol="falcon"
    )
    y = torch.tensor([[[[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4], [4, 5, 6, 7]]]]).share(
        bob, alice, james, protocol="falcon"
    )
    assert (
        falcon.conv2d(x, y, padding=1).reconstruct()
        == torch.tensor([[[[123, 180, 132], [224, 312, 224], [132, 180, 123]]]])
    ).all()


def test_private_compare(workers):
    l = 2 ** 5
    L = 2 ** l
    p = 37

    players = (workers["alice"], workers["bob"], workers["james"])
    x_bit_sh = (
        FalconHelper.decompose(torch.LongTensor([13]), L).share(*players, protocol="falcon").child
    )
    r_bit = (
        FalconHelper.decompose(torch.LongTensor([12]), L).share(*players, protocol="falcon").child
    )

    beta_b = torch.LongTensor([0]).share(
        *players, protocol="falcon", field=2, **FalconHelper.no_wrap
    )
    m = torch.randint(low=1, high=p, size=[1]).share(
        *players, protocol="falcon", field=p, **FalconHelper.no_wrap
    )

    beta_p = falcon.private_compare(x_bit_sh, r_bit, beta_b, m).reconstruct()
    assert beta_p

    beta_b = torch.LongTensor([1]).share(
        *players, protocol="falcon", field=2, **FalconHelper.no_wrap
    )
    m = torch.randint(low=1, high=p, size=[1]).share(
        *players, protocol="falcon", field=p, **FalconHelper.no_wrap
    )

    beta_p = falcon.private_compare(x_bit_sh, r_bit, beta_b, m).reconstruct()
    assert beta_p

    # Big values
    x_bit_sh = (
        FalconHelper.decompose(torch.LongTensor([2 ** 60]), L)
        .share(*players, protocol="falcon")
        .child
    )
    r_bit = (
        FalconHelper.decompose(torch.LongTensor([2 ** 61]), L)
        .share(*players, protocol="falcon")
        .child
    )

    beta_b = torch.LongTensor([0]).share(
        *players, protocol="falcon", field=2, **FalconHelper.no_wrap
    )
    m = torch.randint(low=1, high=p, size=[1]).share(
        *players, protocol="falcon", field=p, **FalconHelper.no_wrap
    )

    beta_p = falcon.private_compare(x_bit_sh, r_bit, beta_b, m).reconstruct()
    assert not beta_p

    beta_b = torch.LongTensor([1]).share(
        *players, protocol="falcon", field=2, **FalconHelper.no_wrap
    )
    m = torch.randint(low=1, high=p, size=[1]).share(
        *players, protocol="falcon", field=p, **FalconHelper.no_wrap
    )
    beta_p = falcon.private_compare(x_bit_sh, r_bit, beta_b, m).reconstruct()
    assert not beta_p

    # Multidimensional tensors
    x_bit_sh = (
        FalconHelper.decompose(torch.LongTensor([[13, 44], [1, 28]]), L)
        .share(*players, protocol="falcon")
        .child
    )
    r_bit = (
        FalconHelper.decompose(torch.LongTensor([[12, 44], [12, 33]]), L)
        .share(*players, protocol="falcon")
        .child
    )

    beta_b = torch.LongTensor([1]).share(
        *players, protocol="falcon", field=2, **FalconHelper.no_wrap
    )
    m = torch.randint(low=1, high=p, size=[1]).share(
        *players, protocol="falcon", field=p, **FalconHelper.no_wrap
    )
    beta_p = falcon.private_compare(x_bit_sh, r_bit, beta_b, m).reconstruct()
    assert (beta_p == torch.tensor([[0, 1], [1, 1]])).all()

    beta_b = torch.LongTensor([0]).share(
        *players, protocol="falcon", field=2, **FalconHelper.no_wrap
    )
    m = torch.randint(low=1, high=p, size=[1]).share(
        *players, protocol="falcon", field=p, **FalconHelper.no_wrap
    )
    beta_p = falcon.private_compare(x_bit_sh, r_bit, beta_b, m).reconstruct()
    assert (beta_p == torch.tensor([[1, 0], [0, 0]])).all()
