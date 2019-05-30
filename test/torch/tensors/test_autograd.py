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


def test_add_backwards(workers):

    alice = workers["alice"]
    a = torch.tensor([3, 2.0, 0], requires_grad=True)
    b = torch.tensor([1, 2.0, 3], requires_grad=True)

    a = a.send(alice, local_autograd=True)
    b = b.send(alice, local_autograd=True)

    a_torch = torch.tensor([3, 2.0, 0], requires_grad=True)
    b_torch = torch.tensor([1, 2.0, 3], requires_grad=True)

    c = a + b
    c_torch = a_torch + b_torch

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.get() == a_torch.grad).all()
    assert (b.grad.get() == b_torch.grad).all()


def test_mul_backwards(workers):

    alice = workers["alice"]
    a = torch.tensor([3, 2.0, 0], requires_grad=True)
    b = torch.tensor([1, 2.0, 3], requires_grad=True)

    a = a.send(alice, local_autograd=True)
    b = b.send(alice, local_autograd=True)

    a_torch = torch.tensor([3, 2.0, 0], requires_grad=True)
    b_torch = torch.tensor([1, 2.0, 3], requires_grad=True)

    c = a * b
    c_torch = a_torch * b_torch

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.get() == a_torch.grad).all()
    assert (b.grad.get() == b_torch.grad).all()


def test_sqrt_backwards(workers):
    alice = workers["alice"]

    a = torch.tensor([3, 2.0, 0], requires_grad=True)
    a = a.send(alice, local_autograd=True)

    a_torch = torch.tensor([3, 2.0, 0], requires_grad=True)

    c = a.sqrt()
    c_torch = a_torch.sqrt()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (a.grad.get() == a_torch.grad).all()


def test_asin_backwards(workers):
    alice = workers["alice"]

    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    a = a.send(alice, local_autograd=True)

    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = a.asin()
    c_torch = a_torch.asin()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (a.grad.get() == a_torch.grad).all()


def test_sin_backwards(workers):
    alice = workers["alice"]

    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    a = a.send(alice, local_autograd=True)

    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = a.sin()
    c_torch = a_torch.sin()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (a.grad.get() == a_torch.grad).all()


def test_sinh_backwards(workers):
    alice = workers["alice"]

    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    a = a.send(alice, local_autograd=True)

    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = a.sinh()
    c_torch = a_torch.sinh()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (a.grad.get() == a_torch.grad).all()


def test_tanh_backwards(workers):
    alice = workers["alice"]

    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    a = a.send(alice, local_autograd=True)

    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = a.tanh()
    c_torch = a_torch.tanh()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (a.grad.get() == a_torch.grad).all()


def test_sigmoid_backwards(workers):
    alice = workers["alice"]

    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    a = a.send(alice, local_autograd=True)

    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = a.sigmoid()
    c_torch = a_torch.sigmoid()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (a.grad.get() == a_torch.grad).all()


# def test_multi_add_sigmoid_backwards(workers):
#     alice = workers["alice"]

#     a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
#     b = torch.tensor([1, 2.0, 3], requires_grad=True)
#     c = torch.tensor([-1, -1, 2.0], requires_grad=True)

#     a = a.send(alice, local_autograd=True)
#     b = b.send(alice, local_autograd=True)
#     c = c.send(alice, local_autograd=True)

#     a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)
#     b_torch = torch.tensor([1, 2.0, 3], requires_grad=True)
#     c_torch = torch.tensor([-1, -1, 2.0], requires_grad=True)

#     res = ((a * b) + c).sigmoid()
#     res = ((a_torch * b_torch) + c_torch).sigmoid()

#     res.backward(torch.ones(res.shape).send(alice))
#     res_torch.backward(torch.ones_like(res_torch))

#     # Have to do .child.grad here because .grad doesn't work on Wrappers yet
#     assert (a.grad.get() == a_torch.grad).all()
