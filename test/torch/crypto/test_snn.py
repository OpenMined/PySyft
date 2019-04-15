import torch
import torch as th
import syft

from syft.frameworks.torch.crypto.securenn import relu_deriv, private_compare, relu, decompose


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


def test_relu(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = th.tensor([1, -3]).share(alice, bob, crypto_provider=james)
    r = x.relu()

    assert (r.get() == th.tensor([1, 0])).all()

    x = th.tensor([1.0, 3.1, -2.1]).fix_prec().share(alice, bob, crypto_provider=james)
    r = x.relu()

    assert (r.get().float_prec() == th.tensor([1, 3.1, 0])).all()
