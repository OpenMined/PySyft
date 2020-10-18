import syft
import torch


# interface
def test_sharing(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    plain_text = torch.tensor([3, 7, 11])
    secret = plain_text.share(bob, alice, james, protocol="falcon")
    assert isinstance(secret.child, syft.ReplicatedSharingTensor)
    assert type(secret.child.child) == dict


def test_reconstruction(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    plain_text = torch.tensor([3, -7, 11])
    secret = plain_text.share(bob, alice, james, protocol="falcon", field=2 ** 5)
    decryption = secret.reconstruct()
    assert (plain_text == decryption).all()


def test_private_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = torch.tensor([-2, 5]).share(bob, alice, james, protocol="falcon")
    assert torch.allclose(x.add(y).reconstruct(), torch.tensor([5, 9]))


def test_public_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = torch.Tensor([-2, 5])
    assert torch.allclose(x.add(y).reconstruct(), torch.tensor([5, 9]))


def test_reversed_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = 1
    assert torch.allclose((y + x).reconstruct(), torch.tensor([8, 5]))


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


def test_reversed_sub(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = 1
    assert torch.allclose((y - x).reconstruct(), torch.tensor([-6, -3]))


def test_add_with_operator(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, 4]).share(bob, alice, james, protocol="falcon")
    y = torch.Tensor([2, 5])
    assert torch.allclose((x + y).reconstruct(), torch.tensor([9, 9]))


def test_public_mul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, -4]).share(bob, alice, james, protocol="falcon")
    assert torch.allclose((x * 2).reconstruct(), torch.tensor([14, -8]))


def test_reversed_mul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([7, -4]).share(bob, alice, james, protocol="falcon")
    assert torch.allclose((2 * x).reconstruct(), torch.tensor([14, -8]))


def test_private_mul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([3, -5]).share(bob, alice, james, protocol="falcon")
    y = torch.tensor([5, -2]).share(bob, alice, james, protocol="falcon")
    assert torch.allclose((x * y).reconstruct(), torch.tensor([15, 10]))


def test_public_matmul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([[1, 2], [3, 4]]).share(bob, alice, james, protocol="falcon")
    y = torch.tensor([[1, 2], [1, 2]])
    assert torch.allclose((x.matmul(y)).reconstruct(), torch.tensor([[3, 6], [7, 14]]))


def test_private_matmul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([[1, 2], [3, 4]]).share(bob, alice, james, protocol="falcon")
    y = torch.tensor([[1, 2], [1, 2]]).share(bob, alice, james, protocol="falcon")
    assert torch.allclose((x.matmul(y)).reconstruct(), torch.tensor([[3, 6], [7, 14]]))


def test_get_shape(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([[1, 2], [3, 4]])
    shape = x.shape
    x = x.share(bob, alice, james, protocol="falcon")
    enc_shape = x.shape
    assert shape == enc_shape


def test_get_players(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([[1, 2], [3, 4]]).share(bob, alice, james, protocol="falcon")
    assert x.players == [bob, alice, james]


def test_view(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.rand([2, 1]).long().share(bob, alice, james, protocol="falcon")
    x = x.view([1, 2])
    assert x.shape == torch.Size([1, 2])


# corner cases
def test_consecutive_arithmetic(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([1, 2]).share(bob, alice, james, protocol="falcon")
    y = torch.tensor([1, 2]).share(bob, alice, james, protocol="falcon")
    z = x * x + y * 2 - x * 4
    assert torch.allclose(z.reconstruct(), torch.tensor([-1, 0]))


def test_negative_result(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor(7).share(bob, alice, james, protocol="falcon")
    y = torch.tensor(3).share(bob, alice, james, protocol="falcon")
    assert y.sub(x).reconstruct() == -4


# utility functions
def test_shares_number():
    tensor = syft.ReplicatedSharingTensor()
    secret = torch.tensor(7)
    number_of_shares = 4
    shares = tensor.generate_shares(secret, number_of_shares)
    assert len(shares) == number_of_shares


def test_workers_arrangement(workers):
    me, bob, alice = (workers["me"], workers["bob"], workers["alice"])
    x = torch.tensor(7).share(bob, alice, me, protocol="falcon")
    assert x.players[0] == me


def test_fixed_precision_and_sharing(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    t = torch.tensor([3.25, 6.83, 8.21, 5.506])
    x = t.fix_prec().share(bob, alice, james, protocol="falcon")
    out = x.reconstruct().float_prec()
    assert torch.allclose(out, t)


def test_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    # 3 workers
    t = torch.tensor([1, 2, 3])
    x = torch.tensor([1, 2, 3]).share(bob, alice, james, protocol="falcon")
    y = (x + x).reconstruct()
    assert torch.allclose(y, (t + t))

    # negative numbers
    t = torch.tensor([1, -2, 3])
    x = torch.tensor([1, -2, 3]).share(bob, alice, james, protocol="falcon")
    y = (x + x).reconstruct()
    assert torch.allclose(y, (t + t))

    # with fixed precisions
    t = torch.tensor([1.0, -2, 3])
    x = torch.tensor([1.0, -2, 3]).fix_prec().share(bob, alice, james, protocol="falcon")
    y = (x + x).reconstruct().float_prec()
    assert torch.allclose(y, (t + t))

    # with FPT>torch.tensor
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(bob, alice, james, protocol="falcon")
    y = t.fix_prec()
    z = (x + y).reconstruct().float_prec()
    assert torch.allclose(z, (t + t))

    z = (y + x).reconstruct().float_prec()
    assert torch.allclose(z, (t + t))

    # with constant integer
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, james, protocol="falcon")
    c = 4

    z = (x + c).reconstruct().float_prec()
    assert torch.allclose(z, (t + c))

    z = (c + x).reconstruct().float_prec()
    assert torch.allclose(z, (c + t))

    # with constant float
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, james, protocol="falcon")
    c = 4.2
    z = (x + c).reconstruct().float_prec()
    assert torch.allclose(z, (t + c))

    z = (c + x).reconstruct().float_prec()
    assert torch.allclose(z, (c + t))


def test_sub(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    t = torch.tensor([1, 2, 3])
    x = torch.tensor([1, 2, 3]).share(bob, alice, james, protocol="falcon")
    y = (x - x).reconstruct()
    assert torch.allclose(y, (t - t))

    # negative numbers
    t = torch.tensor([1, -2, 3])
    x = torch.tensor([1, -2, 3]).share(bob, alice, james, protocol="falcon")
    y = (x - x).reconstruct()
    assert torch.allclose(y, (t - t))

    # with fixed precision
    t = torch.tensor([1.0, -2, 3])
    x = torch.tensor([1.0, -2, 3]).fix_prec().share(bob, alice, james, protocol="falcon")
    y = (x - x).reconstruct().float_prec()
    assert torch.allclose(y, (t - t))

    # with FPT>torch.tensor
    t = torch.tensor([1.0, -2.0, 3.0])
    u = torch.tensor([4.0, 3.0, 2.0])
    x = t.fix_prec().share(bob, alice, james, protocol="falcon")
    y = u.fix_prec()
    z = (x - y).reconstruct().float_prec()
    assert torch.allclose(z, (t - u))

    z = (y - x).reconstruct().float_prec()
    assert torch.allclose(z, (u - t))

    # with constant integer
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, james, protocol="falcon")
    c = 4
    z = (x - c).reconstruct().float_prec()
    assert torch.allclose(z, (t - c))

    z = (c - x).reconstruct().float_prec()
    assert torch.allclose(z, (c - t))

    # with constant float
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, james, protocol="falcon")
    c = 4.2
    z = (x - c).reconstruct().float_prec()
    assert torch.allclose(z, (t - c))

    z = (c - x).reconstruct().float_prec()
    assert torch.allclose(z, (c - t))
