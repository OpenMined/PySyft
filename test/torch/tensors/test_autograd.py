import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import syft

from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.generic.pointers.pointer_tensor import PointerTensor


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


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__", "__matmul__"])
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


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__", "__matmul__"])
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


@pytest.mark.parametrize("cmd", ["asin", "sin", "sinh", "tanh", "sigmoid"])
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


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__", "__matmul__"])
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


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__", "__matmul__"])
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


def test_share_with_requires_grad(workers):
    """
    Test calling fix_precision and share(requires_grad=True) on tensors and model
    """
    bob, alice, charlie, crypto_provider = (
        workers["bob"],
        workers["alice"],
        workers["charlie"],
        workers["james"],
    )

    t = torch.Tensor([3.0])
    t = t.fix_precision()
    t = t.share(alice, bob, crypto_provider=crypto_provider, requires_grad=True)

    assert t.is_wrapper and isinstance(t.child, AutogradTensor)

    t = t.get()

    assert t.is_wrapper and isinstance(t.child, AutogradTensor)

    t = t.float_prec()

    assert t == torch.Tensor([3.0])


def test_remote_share_with_requires_grad(workers):
    """
    Test calling fix_precision and share(requires_grad=True) on pointers
    to tensors and model
    """
    bob, alice, charlie, crypto_provider = (
        workers["bob"],
        workers["alice"],
        workers["charlie"],
        workers["james"],
    )

    t = torch.Tensor([3])
    t = t.send(charlie)
    t = t.fix_precision()
    t = t.share(alice, bob, crypto_provider=crypto_provider, requires_grad=True)
    t = t.get()

    assert isinstance(t.child, AutogradTensor)

    t = torch.Tensor([3])
    t = t.fix_precision()
    t = t.send(charlie)
    t = t.share(alice, bob, crypto_provider=crypto_provider, requires_grad=True)
    t = t.get()

    assert isinstance(t.child, AutogradTensor)

    model = nn.Linear(2, 1)
    model.send(charlie)
    model.fix_precision()
    model.share(alice, bob, crypto_provider=crypto_provider, requires_grad=True)
    model.get()

    assert isinstance(model.weight.child, AutogradTensor)

    # See Issue #2546

    # model = nn.Linear(2, 1)
    # model.fix_precision()
    # model.send(charlie)
    # model.share(alice, bob, crypto_provider=crypto_provider, requires_grad=True)
    # model.get()
    #
    # assert isinstance(model.weight.child, AutogradTensor)


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


def test_serialize_deserialize_autograd_tensor(workers):
    # let's try to send an autogradTensor to a remote location and get it back
    alice = workers["alice"]

    random_tensor = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    assert isinstance(random_tensor, torch.Tensor)

    remote_tensor = random_tensor.send(alice, local_autograd=True)
    assert isinstance(remote_tensor.child, syft.AutogradTensor)

    local_tensor = remote_tensor.get()
    assert isinstance(local_tensor, torch.Tensor)

    # check if the tensor sent is equal to the tensor got from the remote version
    assert torch.all(torch.eq(local_tensor, random_tensor))


def test_types_auto_remote_tensors(workers):
    alice = workers["alice"]
    bob = workers["bob"]

    random_tensor = torch.tensor([[3.0, 2], [-1, 2]], requires_grad=True)
    assert isinstance(random_tensor, torch.Tensor)

    remote_tensor_auto = random_tensor.send(alice, local_autograd=True)
    assert isinstance(remote_tensor_auto.child, syft.AutogradTensor)

    remote_tensor_remote = remote_tensor_auto.send(bob)
    assert isinstance(remote_tensor_remote, torch.Tensor)

    assert type(remote_tensor_auto) == type(remote_tensor_remote)


def test_train_remote_autograd_tensor(workers):
    # Training procedure to train an input model, be it remote or local

    def train(model_input, data_input, target_input, remote=False):
        opt = optim.SGD(params=model_input.parameters(), lr=0.1)
        loss_previous = 99999999999  # just a very big number
        for iter in range(10):
            # 1) erase previous gradients (if they exist)
            opt.zero_grad()
            # 2) make a prediction
            predictions = model_input(data_input)
            # 3) calculate how much we missed
            loss = ((predictions - target_input) ** 2).sum()
            # check for monotonic decrease of the loss

            if remote:
                # Remote loss monotonic decrease
                loss_val_local = loss.copy().get().item()
                assert loss_val_local < loss_previous
                loss_previous = loss_val_local

            else:
                # Local loss monotonic decrease
                loss_val_local = loss.item()
                assert loss_val_local < loss_previous
                loss_previous = loss_val_local

            # 4) Figure out which weights caused us to miss
            loss.backward()
            # 5) change those weights
            opt.step()
        return (loss, model_input)

    alice = workers["alice"]

    # Some Toy Data
    data_local = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.0]])
    target_local = torch.tensor([[0], [0], [1], [1.0]])

    # Toy local model
    model_local = nn.Linear(2, 1)

    # Local training
    loss_local, model_local_trained = train(model_local, data_local, target_local, remote=False)

    # Remote training, setting autograd for the data and the targets
    data_remote = data_local.send(alice, local_autograd=True)
    assert isinstance(data_remote.child, syft.AutogradTensor)
    assert isinstance(data_remote.child.child, PointerTensor)

    target_remote = target_local.send(alice, local_autograd=True)
    assert isinstance(target_remote.child, syft.AutogradTensor)
    assert isinstance(target_remote.child.child, PointerTensor)

    model_remote = model_local.send(alice, local_autograd=True)
    assert isinstance(model_remote.weight.child, syft.AutogradTensor)
    assert isinstance(model_remote.weight.child.child, PointerTensor)

    assert type(model_remote) == type(model_local)

    loss_remote, model_remote_trained = train(model_remote, data_remote, target_remote, remote=True)

    # let's check if the local version and the remote version have the same weight
    assert torch.all(
        torch.eq(
            model_local_trained.weight.copy().get().data, model_remote.weight.copy().get().data
        )
    )


def test_train_without_requires_grad(workers):
    def train(enc_model, enc_data, enc_target):
        optimizer = torch.optim.SGD(enc_model.parameters(), lr=0.1).fix_precision()

        for i in range(1):
            optimizer.zero_grad()
            enc_pred = enc_model(enc_data).squeeze(1)
            l = (((enc_pred - enc_target) ** 2)).sum().refresh()
            l.backward()
            optimizer.step()

        return enc_model.weight.copy().get().data

    alice = workers["alice"]
    bob = workers["bob"]
    james = workers["james"]

    x_1 = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(torch.float)
    y_1 = torch.tensor([0, 0, 1, 1]).to(torch.float)

    enc_data_1 = x_1.fix_precision().share(alice, bob, crypto_provider=james, requires_grad=True)
    enc_target_1 = y_1.fix_precision().share(alice, bob, crypto_provider=james, requires_grad=True)

    model_1 = torch.nn.Linear(2, 1)
    model_2 = torch.nn.Linear(2, 1)

    # Make sure both networks have the same initial values for the parameters
    for param_model_2, param_model_1 in zip(model_2.parameters(), model_1.parameters()):
        param_model_2.data = param_model_1.data

    enc_model_1 = model_1.fix_precision().share(
        alice, bob, crypto_provider=james, requires_grad=True
    )

    model_weights_1 = train(enc_model_1, enc_data_1, enc_target_1)

    # Prepare for new train
    bob.clear_objects()
    alice.clear_objects()
    james.clear_objects()

    enc_model_2 = model_2.fix_precision().share(
        alice, bob, crypto_provider=james, requires_grad=True
    )

    # Without requires_grad
    x_2 = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(torch.float)
    y_2 = torch.tensor([0, 0, 1, 1]).to(torch.float)

    enc_data_2 = x_2.fix_precision().share(alice, bob, crypto_provider=james)
    enc_target_2 = y_2.fix_precision().share(alice, bob, crypto_provider=james)

    model_weights_2 = train(enc_model_2, enc_data_2, enc_target_2)

    # Check the weights for the two models
    assert torch.all(torch.eq(model_weights_1, model_weights_2))


def test_garbage_collection(workers):
    alice = workers["alice"]
    bob = workers["bob"]
    crypto_provider = workers["james"]

    a = torch.ones(1, 5)
    b = torch.ones(1, 5)
    a = a.encrypt(workers=[alice, bob], crypto_provider=crypto_provider, requires_grad=True)
    b = b.encrypt(workers=[alice, bob], crypto_provider=crypto_provider, requires_grad=True)

    class Classifier(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super(Classifier, self).__init__()
            self.fc = torch.nn.Linear(in_features, out_features)

        def forward(self, x):
            logits = self.fc(x)
            return logits

    classifier = Classifier(in_features=5, out_features=5)
    model = classifier.fix_prec().share(
        alice, bob, crypto_provider=crypto_provider, requires_grad=True
    )
    opt = optim.SGD(params=model.parameters(), lr=0.1).fix_precision()
    num_objs = 11
    prev_loss = float("inf")
    for i in range(3):
        preds = classifier(a)
        loss = ((b - preds) ** 2).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
        loss = loss.get().float_prec()

        assert len(alice.object_store._objects) == num_objs
        assert len(bob.object_store._objects) == num_objs
        assert loss < prev_loss

        prev_loss = loss
