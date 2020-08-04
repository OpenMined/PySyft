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
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = torch.tensor([2, 5]).share(bob, alice, james, protocol="falcon")
    assert (x.add(y).reconstruct() == torch.Tensor([9, 9])).all()


def test_public_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = torch.Tensor([2, 5])
    assert (x.add(y).reconstruct() == torch.Tensor([9, 9])).all()


def test_private_sub(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor(7).share(bob, alice, james, protocol="falcon")
    y = torch.tensor(3).share(bob, alice, james, protocol="falcon")
    assert x.sub(y).reconstruct() == 4


def test_public_sub(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor(7).share(bob, alice, james, protocol="falcon")
    y = 3
    assert x.sub(y).reconstruct() == 4


def test_negative_result(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor(7).share(bob, alice, james, protocol="falcon")
    y = torch.tensor(3).share(bob, alice, james, protocol="falcon")
    assert y.sub(x).reconstruct() == -4


def test_workers_arrangement(workers):
    me, bob, alice = (workers["me"], workers["bob"], workers["alice"])
    x = torch.tensor(7).share(bob, alice, me, protocol="falcon")
    assert x.get_players()[0] == me


def test_add_with_operator(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = torch.Tensor([2, 5])
    assert ((x + y).reconstruct() == torch.Tensor([9, 9])).all()
