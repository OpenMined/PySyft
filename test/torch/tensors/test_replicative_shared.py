import syft
import torch


def test_sharing(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    plain_text = torch.tensor([3, 7, 11])
    secret = plain_text.share(bob, alice, james, protocol="falcon")
    assert type(secret) == syft.ReplicatedSharingTensor
    assert type(secret.child) == dict


def test_reconstruction(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    plain_text = torch.tensor([3, 7, 11])
    secret = plain_text.share(bob, alice, james, protocol="falcon")
    decryption = secret.reconstruct()
    assert (plain_text == decryption).all()
