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


@pytest.mark.parametrize("cmd", ["__add__", "__mul__"])
def test_backward_for_remote_binary_cmd_local_autograd(workers, cmd):

    alice = workers["alice"]
    a = torch.tensor([3, 2.0, 0], requires_grad=True)
    b = torch.tensor([1, 2.0, 3], requires_grad=True)

    a = a.send(alice, local_autograd=True)
    b = b.send(alice, local_autograd=True)

    a_torch = torch.tensor([3, 2.0, 0], requires_grad=True)
    b_torch = torch.tensor([1, 2.0, 3], requires_grad=True)

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
