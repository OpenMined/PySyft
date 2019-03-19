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

def test_add_backwards():
    a1 = AutogradTensor().on(torch.tensor([3, 2., 0]))
    b1 = AutogradTensor().on(torch.tensor([1, 2., 3]))

    a2 = torch.tensor([3, 2., 0], requires_grad=True)
    b2 = torch.tensor([1, 2., 3], requires_grad=True)

    c1 = a1 + b1
    c2 = a2 + b2

    c1.backward(torch.ones_like(c1))
    c2.backward(torch.ones_like(c2))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert torch.all(a1.child.grad == a2.grad)
    assert torch.all(b1.child.grad == b2.grad)

def test_mul_backwards():
    a1 = AutogradTensor().on(torch.tensor([3, 2., 0]))
    b1 = AutogradTensor().on(torch.tensor([1, 2., 3]))

    a2 = torch.tensor([3, 2., 0], requires_grad=True)
    b2 = torch.tensor([1, 2., 3], requires_grad=True)

    c1 = a1 * b1
    c2 = a2 * b2

    c1.backward(torch.ones_like(c1))
    c2.backward(torch.ones_like(c2))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert torch.all(a1.child.grad == a2.grad)
    assert torch.all(b1.child.grad == b2.grad)

def test_sqrt_backwards():
    a1 = AutogradTensor().on(torch.tensor([3, 2., 0]))
    a2 = torch.tensor([3, 2., 0], requires_grad=True)

    c1 = a1.sqrt()
    c2 = a2.sqrt()

    c1.backward(torch.ones_like(c1))
    c2.backward(torch.ones_like(c2))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert torch.all(a1.child.grad == a2.grad)

def test_asin_backwards():
    a1 = AutogradTensor().on(torch.tensor([.3, .2, 0]))
    a2 = torch.tensor([.3, .2, 0], requires_grad=True)

    c1 = a1.asin()
    c2 = a2.asin()

    c1.backward(torch.ones_like(c1))
    c2.backward(torch.ones_like(c2))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert torch.all(a1.child.grad == a2.grad)

def test_sin_backwards():
    a1 = AutogradTensor().on(torch.tensor([3, 2., 0]))
    a2 = torch.tensor([3, 2., 0], requires_grad=True)

    c1 = a1.sin()
    c2 = a2.sin()

    c1.backward(torch.ones_like(c1))
    c2.backward(torch.ones_like(c2))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert torch.all(a1.child.grad == a2.grad)

def test_sinh_backwards():
    a1 = AutogradTensor().on(torch.tensor([3, 2., 0]))
    a2 = torch.tensor([3, 2., 0], requires_grad=True)

    c1 = a1.sinh()
    c2 = a2.sinh()

    c1.backward(torch.ones_like(c1))
    c2.backward(torch.ones_like(c2))
    
    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert torch.all(a1.child.grad == a2.grad)