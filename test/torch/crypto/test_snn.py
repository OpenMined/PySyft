import torch
import torch as th
import syft

from syft.frameworks.torch.crypto.securenn import private_compare, decompose, share_convert


def test_xor_implementation(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    r = decompose(th.tensor([3])).send(alice, bob).child
    x_bit_sh = decompose(th.tensor([23])).share(alice, bob, crypto_provider=james).child
    j0 = torch.zeros(x_bit_sh.shape).long().send(bob)
    j1 = torch.ones(x_bit_sh.shape).long().send(alice)
    j = syft.MultiPointerTensor(children=[j0, j1])
    w = (j * r) + x_bit_sh - (2 * x_bit_sh * r)

    r_real = r.virtual_get()[0]
    x_real = x_bit_sh.virtual_get()
    w_real = r_real + x_real - 2 * r_real * x_real
    assert (w.virtual_get() == w_real).all()


def test_private_compare(workers):
    """
    Test private compare which returns: β′ = β ⊕ (x > r).
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x_bit_sh = decompose(torch.LongTensor([13])).share(alice, bob, crypto_provider=james).child
    r = torch.LongTensor([12]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)
    assert not beta_p

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)

    assert beta_p

    # Big values
    x_bit_sh = decompose(torch.LongTensor([2 ** 60])).share(alice, bob, crypto_provider=james).child
    r = torch.LongTensor([2 ** 61]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)
    assert beta_p

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)

    assert not beta_p

    # Negative values
    x_bit_sh = decompose(torch.LongTensor([-105])).share(alice, bob, crypto_provider=james).child
    r = torch.LongTensor([-52]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)
    assert beta_p

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)

    assert not beta_p


def test_share_convert(workers):
    """
    This is a light test as share_convert is not used for the moment
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x_bit_sh = torch.LongTensor([13]).share(alice, bob, crypto_provider=james).child
    field = x_bit_sh.field

    res = share_convert(x_bit_sh)
    assert res.field == field - 1


def test_relu(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = th.tensor([1, -3]).share(alice, bob, crypto_provider=james)
    r = x.relu()

    assert (r.get() == th.tensor([1, 0])).all()

    x = th.tensor([1.0, 3.1, -2.1]).fix_prec().share(alice, bob, crypto_provider=james)
    r = x.relu()

    assert (r.get().float_prec() == th.tensor([1, 3.1, 0])).all()
