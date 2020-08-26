import torch
import syft.frameworks.torch.mpc.falcon.falcon as falcon


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
