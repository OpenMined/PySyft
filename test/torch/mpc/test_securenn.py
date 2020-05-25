import pytest

import torch

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
    r = decompose(torch.LongTensor([3]), 2 ** 64).send(alice, bob).child
    x_bit_sh = (
        decompose(torch.LongTensor([23]), 2 ** 64)
        .share(alice, bob, crypto_provider=james, dtype="long")
        .child
    )
    j0 = torch.zeros(x_bit_sh.shape).long().send(bob)
    j1 = torch.ones(x_bit_sh.shape).long().send(alice)
    j = MultiPointerTensor(children=[j0.child, j1.child])
    w = (j * r) + x_bit_sh - (2 * x_bit_sh * r)

    r_real = r.virtual_get()[0]
    x_real = x_bit_sh.virtual_get()
    w_real = r_real + x_real - 2 * r_real * x_real
    assert (w.virtual_get() == w_real).all()

    # For dtype int
    r = decompose(torch.IntTensor([3]), 2 ** 32).send(alice, bob).child
    x_bit_sh = (
        decompose(torch.IntTensor([23]), 2 ** 32)
        .share(alice, bob, crypto_provider=james, dtype="int")
        .child
    )
    assert x_bit_sh.field == 2 ** 32 and x_bit_sh.dtype == "int"
    j0 = torch.zeros(x_bit_sh.shape).type(torch.int32).send(bob)
    j1 = torch.ones(x_bit_sh.shape).type(torch.int32).send(alice)
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
    L = 2 ** 64
    x_bit_sh = (
        decompose(torch.LongTensor([13]), L)
        .share(alice, bob, crypto_provider=james, field=67, dtype="custom")
        .child
    )
    r = torch.LongTensor([12]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert not beta_p

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert beta_p

    # Big values
    x_bit_sh = (
        decompose(torch.LongTensor([2 ** 60]), L)
        .share(alice, bob, crypto_provider=james, field=67, dtype="custom")
        .child
    )
    r = torch.LongTensor([2 ** 61]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert beta_p

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert not beta_p

    # Multidimensional tensors
    x_bit_sh = (
        decompose(torch.LongTensor([[13, 44], [1, 28]]), L)
        .share(alice, bob, crypto_provider=james, field=67, dtype="custom")
        .child
    )
    r = torch.LongTensor([[12, 44], [12, 33]]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert (beta_p == torch.tensor([[0, 1], [1, 1]])).all()

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert (beta_p == torch.tensor([[1, 0], [0, 0]])).all()

    # Negative values
    x_val = -105
    r_val = -52 % 2 ** 63  # The protocol works only for values in Zq
    x_bit_sh = (
        decompose(torch.LongTensor([x_val]), L)
        .share(alice, bob, crypto_provider=james, field=67, dtype="custom")
        .child
    )
    r = torch.LongTensor([r_val]).send(alice, bob).child

    beta = torch.LongTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert beta_p

    beta = torch.LongTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert not beta_p

    # With dtype int
    L = 2 ** 32

    x_bit_sh = (
        decompose(torch.IntTensor([13]), L)
        .share(alice, bob, crypto_provider=james, field=67, dtype="custom")
        .child
    )
    r = torch.IntTensor([12]).send(alice, bob).child

    beta = torch.IntTensor([1]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert not beta_p

    beta = torch.IntTensor([0]).send(alice, bob).child
    beta_p = private_compare(x_bit_sh, r, beta, L)
    assert beta_p


def test_share_convert(workers):
    """
    This is a light test as share_convert is not used for the moment
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    L = 2 ** 64
    a_sh = (
        torch.LongTensor([-13, 3567, 2 ** 60])
        .share(alice, bob, crypto_provider=james, field=L)
        .child
    )

    res = share_convert(a_sh)
    assert res.dtype == "custom"
    assert res.field == L - 1
    assert (res.get() == torch.LongTensor([-13, 3567, 2 ** 60])).all()

    # With dtype int
    L = 2 ** 32
    a_sh = (
        torch.IntTensor([13, -3567, 2 ** 30])
        .share(alice, bob, crypto_provider=james, field=L)
        .child
    )

    res = share_convert(a_sh)
    assert res.dtype == "custom"
    assert res.field == L - 1
    assert (res.get() == torch.IntTensor([13, -3567, 2 ** 30])).all()


def test_relu_deriv(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = torch.tensor([10, 0, -3]).share(alice, bob, crypto_provider=james, dtype="long").child
    r = relu_deriv(x)

    assert (r.get() == torch.tensor([1, 1, 0])).all()

    # With dtype int
    x = torch.tensor([10, 0, -3]).share(alice, bob, crypto_provider=james, dtype="int").child
    r = relu_deriv(x)

    assert (r.get() == torch.tensor([1, 1, 0])).all()


def test_relu(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = torch.tensor([1, -3]).share(alice, bob, crypto_provider=james, dtype="long")
    r = x.relu()

    assert (r.get() == torch.tensor([1, 0])).all()

    x = (
        torch.tensor([1.0, 3.1, -2.1])
        .fix_prec(dtype="int")
        .share(alice, bob, crypto_provider=james)
    )
    r = x.relu()

    assert (r.get().float_prec() == torch.tensor([1, 3.1, 0])).all()

    # With dtype int
    x = torch.tensor([1, -3]).share(alice, bob, crypto_provider=james, dtype="int")
    r = x.relu()

    assert (r.get() == torch.tensor([1, 0])).all()


def test_division(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x0 = torch.tensor(10).share(alice, bob, crypto_provider=james, dtype="long").child
    y0 = torch.tensor(2).share(alice, bob, crypto_provider=james, dtype="long").child
    res0 = division(x0, y0, bit_len_max=5)

    x1 = (
        torch.tensor([[25, 9], [10, 30]])
        .share(alice, bob, crypto_provider=james, dtype="long")
        .child
    )
    y1 = (
        torch.tensor([[5, 12], [2, 7]]).share(alice, bob, crypto_provider=james, dtype="long").child
    )
    res1 = division(x1, y1, bit_len_max=5)

    assert res0.get() == torch.tensor(5)
    assert (res1.get() == torch.tensor([[5, 0], [5, 4]])).all()

    # With dtype int
    x0 = torch.tensor(10).share(alice, bob, crypto_provider=james, dtype="int").child
    y0 = torch.tensor(2).share(alice, bob, crypto_provider=james, dtype="int").child
    res0 = division(x0, y0, bit_len_max=5)

    x1 = (
        torch.tensor([[25, 9], [10, 30]])
        .share(alice, bob, crypto_provider=james, dtype="int")
        .child
    )
    y1 = torch.tensor([[5, 12], [2, 7]]).share(alice, bob, crypto_provider=james, dtype="int").child
    res1 = division(x1, y1, bit_len_max=5)

    assert res0.get() == torch.tensor(5)
    assert (res1.get() == torch.tensor([[5, 0], [5, 4]])).all()


def test_maxpool(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = (
        torch.tensor([[10, 0], [15, 7]])
        .share(alice, bob, crypto_provider=james, dtype="long")
        .child
    )
    max, ind = maxpool(x)

    assert max.get() == torch.tensor(15)
    assert ind.get() == torch.tensor(2)

    # With dtype int
    x = torch.tensor([[10, 0], [15, 7]]).share(alice, bob, crypto_provider=james, dtype="int").child
    max, ind = maxpool(x)

    assert max.get() == torch.tensor(15)
    assert ind.get() == torch.tensor(2)


def test_maxpool_deriv(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = (
        torch.tensor([[10, 0], [15, 7]])
        .share(alice, bob, crypto_provider=james, dtype="long")
        .child
    )
    max_d = maxpool_deriv(x)

    assert (max_d.get() == torch.tensor([[0, 0], [1, 0]])).all()

    # With dtype int
    x = torch.tensor([[10, 0], [15, 7]]).share(alice, bob, crypto_provider=james, dtype="int").child
    max_d = maxpool_deriv(x)

    assert (max_d.get() == torch.tensor([[0, 0], [1, 0]])).all()


@pytest.mark.parametrize(
    "kernel_size, stride", [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (3, 3)]
)
def test_maxpool2d(workers, kernel_size, stride):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    def _test_maxpool2d(x):
        x_sh = x.long().share(alice, bob, crypto_provider=james, dtype="long").wrap()
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

    x2 = torch.tensor(
        [
            [[[10, 9.1, 1, 1], [0.72, -2.5, 1, 1], [0.72, -2.5, 1, 1], [0.72, -2.5, 1, 1]]],
            [[[15, 0.6, 1, 1], [1, -3, 1, 1], [1, -3, 1, 1], [1, -3, 1, 1]]],
            [[[1.2, 0.3, 1, 1], [5.5, 6.2, 1, 1], [1, -3, 1, 1], [1, -3, 1, 1]]],
        ]
    )

    _test_maxpool2d(x2)
