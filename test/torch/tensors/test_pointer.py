import random

import torch
import syft

from syft.frameworks.torch.tensors import PointerTensor


def test_init(workers):
    pointer = PointerTensor(id=1000, location=workers["alice"], owner=workers["me"])
    pointer.__str__()


def test_create_pointer(workers):
    x = torch.Tensor([1, 2])
    x.create_pointer()
    x.create_pointer(location=workers["james"])


def test_send_get(workers):
    """Test several send get usages"""
    bob = workers["bob"]
    alice = workers["alice"]

    # simple send
    x = torch.Tensor([1, 2])
    x_ptr = x.send(bob)
    x_back = x_ptr.get()
    assert (x == x_back).all()

    # send with variable overwriting
    x = torch.Tensor([1, 2])
    x = x.send(bob)
    x_back = x.get()
    assert (torch.Tensor([1, 2]) == x_back).all()

    # double send
    x = torch.Tensor([1, 2])
    x_ptr = x.send(bob)
    x_ptr_ptr = x_ptr.send(alice)
    x_ptr_back = x_ptr_ptr.get()
    x_back_back = x_ptr_back.get()
    assert (x == x_back_back).all()

    # double send with variable overwriting
    x = torch.Tensor([1, 2])
    x = x.send(bob)
    x = x.send(alice)
    x = x.get()
    x_back = x.get()
    assert (torch.Tensor([1, 2]) == x_back).all()

    # chained double send
    x = torch.Tensor([1, 2])
    x = x.send(bob).send(alice)
    x_back = x.get().get()
    assert (torch.Tensor([1, 2]) == x_back).all()


def test_repeated_send(workers):
    """Tests that repeated calls to .send(bob) works gracefully.
    Previously garbage collection deleted the remote object
    when .send() was called twice. This test ensures the fix still
    works."""

    # create tensor
    x = torch.Tensor([1, 2])
    print(x.id)

    # send tensor to bob
    x_ptr = x.send(workers["bob"])

    # send tensor again
    x_ptr = x.send(workers["bob"])

    # ensure bob has tensor
    assert x.id in workers["bob"]._objects


def test_remote_autograd(workers):
    """Tests the ability to backpropagate gradients on a remote
    worker."""

    # TEST: simple remote grad calculation

    # create a tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)

    # send tensor to bob
    x = x.send(workers["bob"])

    # do some calculation
    y = (x + x).sum()

    # send gradient to backprop to Bob
    grad = torch.tensor([1.0]).send(workers["bob"])

    # backpropagate on remote machine
    y.backward(grad)

    # check that remote gradient is correct
    xgrad = workers["bob"]._objects[x.id_at_location].grad
    xgrad_target = torch.ones(4).float() + 1
    assert (xgrad == xgrad_target).all()

    # TEST: Ensure remote grad calculation gets properly serded

    # create tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True).send(workers["bob"])

    # create output gradient
    out_grad = torch.tensor([1.0]).send(workers["bob"])

    # compute function
    y = x.sum()

    # backpropagate
    y.backward(out_grad)

    # get the gradient created from backpropagation manually
    x_grad = workers["bob"]._objects[x.id_at_location].grad

    # get the entire x tensor (should bring the grad too)
    x = x.get()

    # make sure that the grads match
    assert (x.grad == x_grad).all()


def test_gradient_send_recv(workers):
    """Tests that gradients are properly sent and received along
    with their tensors."""

    # create a tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)

    # create gradient on tensor
    x.sum().backward(torch.ones(1))

    # save gradient
    orig_grad = x.grad

    # send and get back
    t = x.send(workers["bob"]).get()

    # check that gradient was properly serde
    assert (t.grad == orig_grad).all()
