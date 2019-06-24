import pytest
import torch
import syft

from syft.frameworks.torch.tensors.interpreters import AutogradTensor


def test_wrap():
    """
    Test the .on() wrap functionality for AutogradTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = AutogradTensor().on(x_tensor)

    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, AutogradTensor)
    assert isinstance(x.child.child, torch.Tensor)


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__", "__matmul__"])
def test_backward_for_binary_cmd_with_autograd(cmd):
    """
    Test .backward() on local tensors wrapped in an AutogradTensor
    (It is useless but this is the most basic example)
    """
    a = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    c = getattr(a, cmd)(b)
    c_torch = getattr(a_torch, cmd)(b_torch)

    ones = torch.ones(c.shape)
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.child.grad == a_torch.grad).all()
    assert (b.child.grad == b_torch.grad).all()


@pytest.mark.parametrize("cmd", ["__iadd__", "__isub__"])
def test_backward_for_inplace_binary_cmd_with_autograd(cmd):

    a = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)
    c = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)
    c = syft.AutogradTensor().on(c)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)
    c_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    r = a * b
    getattr(r, cmd)(c)
    r_torch = a_torch * b_torch
    getattr(r_torch, cmd)(c_torch)

    ones = torch.ones(r.shape)
    ones = syft.AutogradTensor().on(ones)
    r.backward(ones)
    r_torch.backward(torch.ones(r_torch.shape))

    assert (a.child.grad == a_torch.grad).all()
    assert (b.child.grad == b_torch.grad).all()
    assert (c.child.grad == c_torch.grad).all()


@pytest.mark.parametrize("cmd", ["__add__", "__sub__"])
@pytest.mark.parametrize(
    "shapes",
    [
        ((2,), (2,)),
        ((5, 3, 2), (5, 1, 2)),
        ((3, 2), (5, 3, 2)),
        ((3, 1), (5, 3, 2)),
        ((3, 2), (5, 1, 2)),
    ],
)
def test_backward_for_binary_cmd_with_inputs_of_different_dim_and_autograd(cmd, shapes):
    """
    Test .backward() on local tensors wrapped in an AutogradTensor
    (It is useless but this is the most basic example)
    """
    a_shape, b_shape = shapes
    a = torch.ones(a_shape, requires_grad=True)
    b = torch.ones(b_shape, requires_grad=True)

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)

    a_torch = torch.ones(a_shape, requires_grad=True)
    b_torch = torch.ones(b_shape, requires_grad=True)

    c = getattr(a, cmd)(b)
    c_torch = getattr(a_torch, cmd)(b_torch)

    ones = torch.ones(c.shape)
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.child.grad == a_torch.grad).all()
    assert (b.child.grad == b_torch.grad).all()


@pytest.mark.parametrize("cmd", ["__add__", "__mul__", "__matmul__"])
def test_backward_for_remote_binary_cmd_with_autograd(workers, cmd):
    """
    Test .backward() on remote tensors using explicit wrapping
    with an Autograd Tensor.
    """
    alice = workers["alice"]

    a = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True).send(alice)
    b = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True).send(alice)

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    c = getattr(a, cmd)(b)
    c_torch = getattr(a_torch, cmd)(b_torch)

    ones = torch.ones(c.shape).send(alice)
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.get() == a_torch.grad).all()
    assert (b.grad.get() == b_torch.grad).all()


@pytest.mark.parametrize("cmd", ["__iadd__", "__isub__"])
def test_backward_for_remote_inplace_binary_cmd_with_autograd(workers, cmd):
    alice = workers["alice"]

    a = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True).send(alice)
    b = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True).send(alice)
    c = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True).send(alice)

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)
    c = syft.AutogradTensor().on(c)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)
    c_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    r = a * b
    getattr(r, cmd)(c)
    r_torch = a_torch * b_torch
    getattr(r_torch, cmd)(c_torch)

    ones = torch.ones(r.shape).send(alice)
    ones = syft.AutogradTensor().on(ones)
    r.backward(ones)
    r_torch.backward(torch.ones(r_torch.shape))

    assert (a.grad.get() == a_torch.grad).all()
    assert (b.grad.get() == b_torch.grad).all()
    assert (c.grad.get() == c_torch.grad).all()


@pytest.mark.parametrize("cmd", ["__add__", "__mul__", "__matmul__"])
def test_backward_for_remote_binary_cmd_local_autograd(workers, cmd):
    """
    Test .backward() on remote tensors using implicit wrapping
    with an Autograd Tensor.

    Distinguish the current use of:
        a = torch.tensor([[3., 2], [-1, 2]], requires_grad=True)
        a.send(alice, local_autograd=True)

    instead of the previous:
        a = torch.tensor([[3., 2], [-1, 2]], requires_grad=True).send(alice)
        a = syft.AutogradTensor().on(a)
    """
    alice = workers["alice"]

    a = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    a = a.send(alice, local_autograd=True)
    b = b.send(alice, local_autograd=True)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    c = getattr(a, cmd)(b)
    c_torch = getattr(a_torch, cmd)(b_torch)

    ones = torch.ones(c.shape).send(alice)
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.get() == a_torch.grad).all()
    assert (b.grad.get() == b_torch.grad).all()


@pytest.mark.parametrize("cmd", ["sqrt", "asin", "sin", "sinh", "tanh", "sigmoid"])
def test_backward_for_remote_unary_cmd_local_autograd(workers, cmd):
    """
    Test .backward() on unary methods on remote tensors using
    implicit wrapping
    """
    alice = workers["alice"]

    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    a = a.send(alice, local_autograd=True)

    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = getattr(a, cmd)()
    c_torch = getattr(a_torch, cmd)()

    ones = torch.ones(c.shape).send(alice)
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones)
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (a.grad.get() == a_torch.grad).all()


@pytest.mark.parametrize("cmd", ["__add__", "__mul__", "__matmul__"])
def test_backward_for_fix_prec_binary_cmd_with_autograd(cmd):
    """
    Test .backward() on Fixed Precision Tensor for a single operation
    """
    a = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True).fix_prec()
    b = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True).fix_prec()

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    c = getattr(a, cmd)(b)
    c_torch = getattr(a_torch, cmd)(b_torch)

    ones = torch.ones(c.shape).fix_prec()
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.float_prec() == a_torch.grad).all()
    assert (b.grad.float_prec() == b_torch.grad).all()


@pytest.mark.parametrize("cmd", ["__add__", "__mul__", "__matmul__"])
def test_backward_for_additive_shared_binary_cmd_with_autograd(workers, cmd):
    """
    Test .backward() on Additive Shared Tensor for a single operation
    """
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]

    a = (
        torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
        .fix_prec()
        .share(alice, bob, crypto_provider=james)
    )
    b = (
        torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)
        .fix_prec()
        .share(alice, bob, crypto_provider=james)
    )

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)

    c = getattr(a, cmd)(b)
    c_torch = getattr(a_torch, cmd)(b_torch)

    ones = torch.ones(c.shape).fix_prec().share(alice, bob, crypto_provider=james)
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.get().float_prec() == a_torch.grad).all()
    assert (b.grad.get().float_prec() == b_torch.grad).all()


def test_addmm_backward_for_additive_shared_with_autograd(workers):
    """
    Test .backward() on Additive Shared Tensor for addmm
    """
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]

    a = (
        torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
        .fix_prec()
        .share(alice, bob, crypto_provider=james)
    )
    b = (
        torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)
        .fix_prec()
        .share(alice, bob, crypto_provider=james)
    )
    c = (
        torch.tensor([[2.0, 2], [0, 1]], requires_grad=True)
        .fix_prec()
        .share(alice, bob, crypto_provider=james)
    )

    a = syft.AutogradTensor().on(a)
    b = syft.AutogradTensor().on(b)
    c = syft.AutogradTensor().on(c)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = torch.tensor([[1.0, 2], [3, 2]], requires_grad=True)
    c_torch = torch.tensor([[2.0, 2], [0, 1]], requires_grad=True)

    r = torch.addmm(c, a, b)
    r_torch = torch.addmm(c_torch, a_torch, b_torch)

    ones = torch.ones(r.shape).fix_prec().share(alice, bob, crypto_provider=james)
    ones = syft.AutogradTensor().on(ones)
    r.backward(ones)
    r_torch.backward(torch.ones(r_torch.shape))

    assert (a.grad.get().float_prec() == a_torch.grad).all()
    assert (b.grad.get().float_prec() == b_torch.grad).all()
    assert (c.grad.get().float_prec() == c_torch.grad).all()


# def test_multi_add_sigmoid_backwards(workers):
#     alice = workers["alice"]

#     a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
#     b = torch.tensor([[1., 2], [3, 2]], requires_grad=True)
#     c = torch.tensor([-1, -1, 2.0], requires_grad=True)

#     a = a.send(alice, local_autograd=True)
#     b = b.send(alice, local_autograd=True)
#     c = c.send(alice, local_autograd=True)

#     a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)
#     b_torch = torch.tensor([[1., 2], [3, 2]], requires_grad=True)
#     c_torch = torch.tensor([-1, -1, 2.0], requires_grad=True)

#     res = ((a * b) + c).sigmoid()
#     res = ((a_torch * b_torch) + c_torch).sigmoid()

#     res.backward(torch.ones(res.shape).send(alice))
#     res_torch.backward(torch.ones_like(res_torch))

#     # Have to do .child.grad here because .grad doesn't work on Wrappers yet
#     assert (a.grad.get() == a_torch.grad).all()
