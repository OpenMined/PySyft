import pytest

import torch
import torch as th

import syft
from syft.frameworks.torch.mpc.securenn import (
    private_compare,
    decompose,
    share_convert,
    relu_deriv,
    division,
    maxpool,
    maxpool2d,
    maxpool_deriv,
)
from syft.generic.pointers.multi_pointer import MultiPointerTensor


def test_xor_implementation(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    r = decompose(th.tensor([3])).send(alice, bob).child
    x_bit_sh = decompose(th.tensor([23])).share(alice, bob, crypto_provider=james).child
    j0 = torch.zeros(x_bit_sh.shape).long().send(bob)
    j1 = torch.ones(x_bit_sh.shape).long().send(alice)
    j = MultiPointerTensor(children=[j0.child, j1.child])
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

    # Multidimensional tensors
    x_bit_sh = (
        decompose(torch.LongTensor([[13, 44], [1, 28]]))
        .share(alice, bob, crypto_provider=james)
        .child
    )
    r = torch.LongTensor([[12, 44], [12, 33]]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)
    assert (beta_p == torch.tensor([[0, 1], [1, 1]])).all()

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta)
    assert (beta_p == torch.tensor([[1, 0], [0, 0]])).all()

    # Negative values
    x_val = -105 % 2 ** 62
    r_val = -52 % 2 ** 62  # The protocol works only for values in Zq
    x_bit_sh = decompose(torch.LongTensor([x_val])).share(alice, bob, crypto_provider=james).child
    r = torch.LongTensor([r_val]).send(alice, bob).child

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
    L = 2 ** 62
    x_bit_sh = (
        torch.LongTensor([13, 3567, 2 ** 60])
        .share(alice, bob, crypto_provider=james, field=L)
        .child
    )

    res = share_convert(x_bit_sh)
    assert res.field == L - 1
    assert (res.get() % L == torch.LongTensor([13, 3567, 2 ** 60])).all()


def test_relu_deriv(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = th.tensor([10, 0, -3]).share(alice, bob, crypto_provider=james).child
    r = relu_deriv(x)

    assert (r.get() == th.tensor([1, 1, 0])).all()


def test_relu(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = th.tensor([1, -3]).share(alice, bob, crypto_provider=james)
    r = x.relu()

    assert (r.get() == th.tensor([1, 0])).all()

    x = th.tensor([1.0, 3.1, -2.1]).fix_prec().share(alice, bob, crypto_provider=james)
    r = x.relu()

    assert (r.get().float_prec() == th.tensor([1, 3.1, 0])).all()


def test_division(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x0 = th.tensor(10).share(alice, bob, crypto_provider=james).child
    y0 = th.tensor(2).share(alice, bob, crypto_provider=james).child
    res0 = division(x0, y0, bit_len_max=5)

    x1 = th.tensor([[25, 9], [10, 30]]).share(alice, bob, crypto_provider=james).child
    y1 = th.tensor([[5, 12], [2, 7]]).share(alice, bob, crypto_provider=james).child
    res1 = division(x1, y1, bit_len_max=5)

    assert res0.get() == torch.tensor(5)
    assert (res1.get() == torch.tensor([[5, 0], [5, 4]])).all()


def test_maxpool(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = th.tensor([[10, 0], [15, 7]]).share(alice, bob, crypto_provider=james).child
    max, ind = maxpool(x)

    assert max.get() == torch.tensor(15)
    assert ind.get() == torch.tensor(2)


def test_maxpool_deriv(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = th.tensor([[10, 0], [15, 7]]).share(alice, bob, crypto_provider=james).child
    max_d = maxpool_deriv(x)

    assert (max_d.get() == torch.tensor([[0, 0], [1, 0]])).all()


@pytest.mark.parametrize(
    "kernel_size, stride", [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (3, 3)]
)
def test_maxpool2d(workers, kernel_size, stride):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    def _test_maxpool2d(x):
        x_sh = x.long().share(alice, bob, crypto_provider=james).wrap()
        y = maxpool2d(x_sh, kernel_size=kernel_size, stride=stride)

        torch_maxpool = torch.nn.MaxPool2d(kernel_size, stride=stride)
        assert torch.all(torch.eq(y.get(), torch_maxpool(x).long()))

    x1 = torch.Tensor(
        [
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ]
    )

    _test_maxpool2d(x1)

    x2 = th.tensor(
        [
            [[[10, 9.1, 1, 1], [0.72, -2.5, 1, 1], [0.72, -2.5, 1, 1], [0.72, -2.5, 1, 1]]],
            [[[15, 0.6, 1, 1], [1, -3, 1, 1], [1, -3, 1, 1], [1, -3, 1, 1]]],
            [[[1.2, 0.3, 1, 1], [5.5, 6.2, 1, 1], [1, -3, 1, 1], [1, -3, 1, 1]]],
        ]
    )

    _test_maxpool2d(x2)
