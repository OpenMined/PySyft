import torch

from syft.frameworks.torch.mpc.aby3.aby3_helper import ABY3Helper


def test_bit_inject(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    plain_text = torch.tensor([0, 1, 0, 1])
    secret = plain_text.share(bob, alice, james, protocol="falcon", field=2)

    new_secret = ABY3Helper.bit_inject(secret)

    assert (new_secret.reconstruct() == plain_text).all()
