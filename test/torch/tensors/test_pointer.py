import random

import torch
import torch as th
import syft

from syft.frameworks.torch.tensors.interpreters import PointerTensor


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


def test_inplace_send_get(workers):
    tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
    tensor_ptr = tensor.send_(workers["bob"])

    assert tensor_ptr.id == tensor.id
    assert id(tensor_ptr) == id(tensor)

    tensor_back = tensor_ptr.get_()

    assert tensor_back.id == tensor_ptr.id
    assert tensor_back.id == tensor.id
    assert id(tensor_back) == id(tensor)
    assert id(tensor_back) == id(tensor)

    assert (tensor_back == tensor).all()


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

    # backpropagate on remote machine
    y.backward()

    # check that remote gradient is correct
    xgrad = workers["bob"]._objects[x.id_at_location].grad
    xgrad_target = torch.ones(4).float() + 1
    assert (xgrad == xgrad_target).all()

    # TEST: Ensure remote grad calculation gets properly serded

    # create tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True).send(workers["bob"])

    # compute function
    y = x.sum()

    # backpropagate
    y.backward()

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


def test_method_on_attribute(workers):

    # create remote object with children
    x = torch.Tensor([1, 2, 3])
    x = syft.LoggingTensor().on(x).send(workers["bob"])

    # call method on data tensor directly
    x.child.point_to_attr = "child.child"
    y = x.add(x)
    assert isinstance(y.get(), torch.Tensor)

    # call method on loggingtensor directly
    x.child.point_to_attr = "child"
    y = x.add(x)
    y = y.get()
    assert isinstance(y.child, syft.LoggingTensor)

    # # call method on zeroth attribute
    # x.child.point_to_attr = ""
    # y = x.add(x)
    # y = y.get()
    #
    # assert isinstance(y, torch.Tensor)
    # assert isinstance(y.child, syft.LoggingTensor)
    # assert isinstance(y.child.child, torch.Tensor)

    # call .get() on pinter to attribute (should error)
    x.child.point_to_attr = "child"
    try:
        x.get()
    except syft.exceptions.CannotRequestTensorAttribute as e:
        assert True


def test_grad_pointer(workers):
    """Tests the automatic creation of a .grad pointer when
    calling .send() on a tensor with requires_grad==True"""

    bob = workers["bob"]

    x = torch.tensor([1, 2, 3.0], requires_grad=True).send(bob)
    y = (x + x).sum()
    y.backward()

    assert (bob._objects[x.id_at_location].grad == torch.tensor([2, 2, 2.0])).all()


def test_move(workers):

    x = torch.tensor([1, 2, 3, 4, 5]).send(workers["bob"])

    assert x.id_at_location in workers["bob"]._objects
    assert x.id_at_location not in workers["alice"]._objects

    x.move(workers["alice"])

    assert x.id_at_location not in workers["bob"]._objects
    assert x.id_at_location in workers["alice"]._objects

    x = torch.tensor([1.0, 2, 3, 4, 5], requires_grad=True).send(workers["bob"])

    assert x.id_at_location in workers["bob"]._objects
    assert x.id_at_location not in workers["alice"]._objects

    x.move(workers["alice"])

    assert x.id_at_location not in workers["bob"]._objects
    assert x.id_at_location in workers["alice"]._objects


def test_combine_pointers(workers):
    """
    Ensure that the sy.combine_pointers works as expected
    """

    bob = workers["bob"]
    alice = workers["alice"]

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    y = th.tensor([1, 2, 3, 4, 5]).send(alice)

    a = x.combine(y)
    b = a + a

    c = b.get(sum_results=True)
    assert (c == th.tensor([4, 8, 12, 16, 20])).all()

    b = a + a
    c = b.get(sum_results=False)
    assert len(c) == 2
    assert (c[0] == th.tensor([2, 4, 6, 8, 10])).all


def test_remote_to_cpu_device(workers):
    """Ensure remote .to cpu works"""
    device = torch.device("cpu")
    bob = workers["bob"]

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    x.to(device)
