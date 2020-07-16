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


def test_shares_number():
    tensor = syft.ReplicatedSharingTensor()
    secret = torch.tensor(7)
    number_of_shares = 4
    shares = tensor.generate_shares(secret, number_of_shares)
    assert len(shares) == number_of_shares


def test_verify_players(workers):
    bob, alice, james, charlie = (
        workers["bob"],
        workers["alice"],
        workers["james"],
        workers["charlie"],
    )
    secret1 = torch.tensor(7).share(bob, alice, james, protocol="falcon")
    secret2 = torch.tensor(5).share(bob, alice, charlie, protocol="falcon")
    assert secret1.verify_matching_players(secret2) is False


def test_private_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor(7).share(bob, alice, james, protocol="falcon")
    y = torch.tensor(3).share(bob, alice, james, protocol="falcon")
    assert x.private_add(y).reconstruct() == 10


def test_public_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor(7).share(bob, alice, james, protocol="falcon")
    y = 3
    assert x.public_add(y).reconstruct() == 10
