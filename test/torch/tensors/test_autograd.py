import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
@pytest.mark.parametrize("backward_one", [True, False])
def test_backward_for_binary_cmd_with_autograd(cmd, backward_one):
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
    c.backward(ones if backward_one else None)
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
@pytest.mark.parametrize("backward_one", [True, False])
def test_backward_for_remote_binary_cmd_with_autograd(workers, cmd, backward_one):
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
    c.backward(ones if backward_one else None)
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
@pytest.mark.parametrize("backward_one", [True, False])
def test_backward_for_fix_prec_binary_cmd_with_autograd(cmd, backward_one):
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
    c.backward(ones if backward_one else None)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.float_prec() == a_torch.grad).all()
    assert (b.grad.float_prec() == b_torch.grad).all()


def test_backward_for_linear_model_on_fix_prec_params_with_autograd():
    """
    Test .backward() on Fixed Precision parameters with mixed operations
    """
    x = torch.tensor([[1.0, 2], [1.0, 2]]).fix_prec()
    target = torch.tensor([[1.0], [1.0]]).fix_prec()
    model = nn.Linear(2, 1)
    model.weight = nn.Parameter(torch.tensor([[-1.0, 2]]))
    model.bias = nn.Parameter(torch.tensor([-1.0]))
    model.fix_precision()

    x = syft.AutogradTensor().on(x)
    target = syft.AutogradTensor().on(target)
    model.weight = syft.AutogradTensor().on(model.weight)
    model.bias = syft.AutogradTensor().on(model.bias)

    output = model(x)
    loss = ((output - target) ** 2).sum()
    one = torch.ones(loss.shape).fix_prec()
    one = syft.AutogradTensor().on(one)
    loss.backward(one)

    weight_grad = model.weight.grad.float_precision()
    bias_grad = model.bias.grad.float_precision()

    x = torch.tensor([[1.0, 2], [1.0, 2]])
    target = torch.tensor([[1.0], [1.0]])
    model = nn.Linear(2, 1)
    model.weight = nn.Parameter(torch.tensor([[-1.0, 2]]))
    model.bias = nn.Parameter(torch.tensor([-1.0]))

    output = model(x)
    loss = ((output - target) ** 2).sum()

    one = torch.ones(loss.shape)
    loss.backward(one)
    assert (model.weight.grad == weight_grad).all()
    assert (model.bias.grad == bias_grad).all()


@pytest.mark.parametrize("cmd", ["__add__", "__mul__", "__matmul__"])
@pytest.mark.parametrize("backward_one", [True, False])
def test_backward_for_additive_shared_binary_cmd_with_autograd(workers, cmd, backward_one):
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
    c.backward(ones if backward_one else None)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.get().float_prec() == a_torch.grad).all()
    assert (b.grad.get().float_prec() == b_torch.grad).all()


@pytest.mark.parametrize("backward_one", [True, False])
def test_backward_for_additive_shared_div_with_autograd(workers, backward_one):
    """
    Test .backward() on Additive Shared Tensor for division with an integer
    """
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]

    a = (
        torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
        .fix_prec()
        .share(alice, bob, crypto_provider=james)
    )
    b = 2

    a = syft.AutogradTensor().on(a)

    a_torch = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    b_torch = 2

    c = a / b
    c_torch = a_torch / b_torch

    ones = torch.ones(c.shape).fix_prec().share(alice, bob, crypto_provider=james)
    ones = syft.AutogradTensor().on(ones)
    c.backward(ones if backward_one else None)
    c_torch.backward(torch.ones(c_torch.shape))

    assert (a.grad.get().float_prec() == a_torch.grad).all()


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


def test_relu_backward_or_additive_shared_with_autograd(workers):
    """
    Test .backward() on Additive Shared Tensor for F.relu
    """
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]
    data = torch.tensor([[-1, -0.1], [1, 0.1]], requires_grad=True)
    data = data.fix_precision().share(bob, alice, crypto_provider=james, requires_grad=True)
    loss = F.relu(data)
    loss.backward()
    expected = torch.tensor([[0.0, 0], [1, 1]])
    assert (data.grad.get().float_prec() == expected).all()


def test_backward_for_linear_model_on_additive_shared_with_autograd(workers):
    """
    Test .backward() on Additive Shared tensors with mixed operations
    """
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]

    x = torch.tensor([[1.0, 2], [1.0, 2]]).fix_prec().share(bob, alice, crypto_provider=james)
    target = torch.tensor([[1.0], [1.0]]).fix_prec().share(bob, alice, crypto_provider=james)
    model = nn.Linear(2, 1)
    model.weight = nn.Parameter(torch.tensor([[-1.0, 2]]))
    model.bias = nn.Parameter(torch.tensor([-1.0]))
    model.fix_precision().share(bob, alice, crypto_provider=james)

    x = syft.AutogradTensor().on(x)
    target = syft.AutogradTensor().on(target)
    model.weight = syft.AutogradTensor().on(model.weight)
    model.bias = syft.AutogradTensor().on(model.bias)

    output = model(x)
    loss = ((output - target) ** 2).sum()
    one = torch.ones(loss.shape).fix_prec().share(bob, alice, crypto_provider=james)
    one = syft.AutogradTensor().on(one)
    loss.backward(one)

    weight_grad = model.weight.grad.get().float_precision()
    bias_grad = model.bias.grad.get().float_precision()

    x = torch.tensor([[1.0, 2], [1.0, 2]])
    target = torch.tensor([[1.0], [1.0]])
    model = nn.Linear(2, 1)
    model.weight = nn.Parameter(torch.tensor([[-1.0, 2]]))
    model.bias = nn.Parameter(torch.tensor([-1.0]))

    output = model(x)
    loss = ((output - target) ** 2).sum()

    one = torch.ones(loss.shape)
    loss.backward(one)
    assert (model.weight.grad == weight_grad).all()
    assert (model.bias.grad == bias_grad).all()


def test_encrypted_training_with_linear_model(workers):
    """
    Test a minimal example of encrypted training using nn.Linear
    """
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]

    # A Toy Dataset
    data = (
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.0]])
        .fix_prec()
        .share(bob, alice, crypto_provider=james)
    )
    target = (
        torch.tensor([[0], [0], [1], [1.0]]).fix_prec().share(bob, alice, crypto_provider=james)
    )

    # A Toy Model
    model = nn.Linear(2, 1).fix_precision().share(bob, alice, crypto_provider=james)

    data = syft.AutogradTensor().on(data)
    target = syft.AutogradTensor().on(target)
    model.weight = syft.AutogradTensor().on(model.weight)
    model.bias = syft.AutogradTensor().on(model.bias)

    def train():
        # Training Logic
        # Convert the learning rate to fixed precision
        opt = optim.SGD(params=model.parameters(), lr=0.1).fix_precision()

        for iter in range(10):

            # 1) erase previous gradients (if they exist)
            opt.zero_grad()

            # 2) make a prediction
            pred = model(data)

            # 3) calculate how much we missed
            loss = ((pred - target) ** 2).sum()

            # 4) figure out which weights caused us to miss
            loss.backward()

            # 5) change those weights
            opt.step()

        return loss

    loss = train()

    assert loss.child.child.child.virtual_get() < 500


def test_get_float_prec_on_autograd_tensor(workers):
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]

    x = torch.tensor([0.1, 1.0])
    x2 = syft.AutogradTensor().on(x.fix_prec())
    assert (x2.float_precision() == x).all()

    x = torch.tensor([1, 2])
    x2 = x.share(bob, alice, crypto_provider=james)
    x2 = syft.AutogradTensor().on(x2)
    assert (x2.get() == x).all()

    x = torch.tensor([0.1, 1.0])
    x2 = x.fix_precision()
    x2 = x2.share(bob, alice, crypto_provider=james, requires_grad=True)
    assert (x2.get().float_precision() == x).all()
