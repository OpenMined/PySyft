import torch
import torch as th
import syft

from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.generic.pointers.pointer_tensor import PointerTensor
import pytest


def test_init(workers):
    alice, me = workers["alice"], workers["me"]
    pointer = PointerTensor(id=1000, location=alice, owner=me)
    pointer.__str__()


def test_create_pointer():
    x = torch.Tensor([1, 2])
    x.create_pointer()


def test_send_default_garbage_collector_true(workers):
    """
    Remote tensor should be garbage collected by default on
    deletion of the Pointer tensor pointing to remote tensor
    """
    alice = workers["alice"]

    x = torch.Tensor([-1, 2])
    x_ptr = x.send(alice)
    assert x_ptr.child.garbage_collect_data


def test_send_garbage_collect_data_false(workers):
    """
    Remote tensor should be not garbage collected on
    deletion of the Pointer tensor pointing to remote tensor
    """
    alice = workers["alice"]

    x = torch.Tensor([-1, 2])
    x_ptr = x.send(alice)
    x_ptr.garbage_collection = False
    assert x_ptr.child.garbage_collect_data == False


def test_send_gc_false(workers):
    """
    Remote tensor should be not garbage collected on
    deletion of the Pointer tensor pointing to remote tensor
    """
    alice = workers["alice"]
    x = torch.Tensor([-1, 2])
    x_ptr = x.send(alice)
    x_ptr.gc = False
    assert x_ptr.child.garbage_collect_data == False
    assert x_ptr.gc == False, "property GC is not in sync"
    assert x_ptr.garbage_collection == False, "property garbage_collection is not in sync"


def test_send_gc_true(workers):
    """
    Remote tensor by default is garbage collected on
    deletion of Pointer Tensor
    """
    alice = workers["alice"]

    x = torch.Tensor([-1, 2])
    x_ptr = x.send(alice)

    assert x_ptr.gc == True


def test_send_disable_gc(workers):
    """Pointer tensor should be not garbage collected."""
    alice = workers["alice"]

    x = torch.Tensor([-1, 2])
    x_ptr = x.send(alice).disable_gc
    assert x_ptr.child.garbage_collect_data == False
    assert x_ptr.gc == False, "property GC is not in sync"
    assert x_ptr.garbage_collection == False, "property garbage_collection is not in sync"


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
    bob = workers["bob"]

    tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
    tensor_ptr = tensor.send_(bob)

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

    bob = workers["bob"]

    # create tensor
    x = torch.Tensor([1, 2])

    # send tensor to bob
    x_ptr = x.send(bob)

    # send tensor again
    x_ptr = x.send(bob)

    # ensure bob has tensor
    assert x.id in bob.object_store._objects


def test_remote_autograd(workers):
    """Tests the ability to backpropagate gradients on a remote
    worker."""

    bob = workers["bob"]

    # TEST: simple remote grad calculation

    # create a tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)

    # send tensor to bob
    x = x.send(bob)

    # do some calculation
    y = (x + x).sum()

    # backpropagate on remote machine
    y.backward()

    # check that remote gradient is correct
    x_grad = bob.object_store.get_obj(x.id_at_location).grad
    x_grad_target = torch.ones(4).float() + 1
    assert (x_grad == x_grad_target).all()

    # TEST: Ensure remote grad calculation gets properly serded

    # create tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True).send(bob)

    # compute function
    y = x.sum()

    # backpropagate
    y.backward()

    # get the gradient created from backpropagation manually
    x_grad = bob.object_store.get_obj(x.id_at_location).grad

    # get the entire x tensor (should bring the grad too)
    x = x.get()

    # make sure that the grads match
    assert (x.grad == x_grad).all()


def test_gradient_send_recv(workers):
    """Tests that gradients are properly sent and received along
    with their tensors."""

    bob = workers["bob"]

    # create a tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)

    # create gradient on tensor
    x.sum().backward(th.tensor(1.0))

    # save gradient
    orig_grad = x.grad

    # send and get back
    t = x.send(bob).get()

    # check that gradient was properly serde
    assert (t.grad == orig_grad).all()


def test_method_on_attribute(workers):

    bob = workers["bob"]

    # create remote object with children
    x = torch.Tensor([1, 2, 3])
    x = syft.LoggingTensor().on(x).send(bob)

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
    except syft.exceptions.CannotRequestObjectAttribute as e:
        assert True


def test_grad_pointer(workers):
    """Tests the automatic creation of a .grad pointer when
    calling .send() on a tensor with requires_grad==True"""

    bob = workers["bob"]

    x = torch.tensor([1, 2, 3.0], requires_grad=True).send(bob)
    y = (x + x).sum()
    y.backward()

    assert (bob.object_store.get_obj(x.id_at_location).grad == torch.tensor([2, 2, 2.0])).all()


def test_move(workers):
    alice, bob, james, me = workers["alice"], workers["bob"], workers["james"], workers["me"]

    x = torch.tensor([1, 2, 3, 4, 5]).send(bob)

    assert x.id_at_location in bob.object_store._objects
    assert x.id_at_location not in alice.object_store._objects

    p = x.move(alice)

    assert x.id_at_location not in bob.object_store._objects
    assert x.id_at_location in alice.object_store._objects

    x = torch.tensor([1.0, 2, 3, 4, 5], requires_grad=True).send(bob)

    assert x.id_at_location in bob.object_store._objects
    assert x.id_at_location not in alice.object_store._objects

    p = x.move(alice)

    assert x.id_at_location not in bob.object_store._objects
    assert x.id_at_location in alice.object_store._objects

    alice.clear_objects()
    bob.clear_objects()
    x = torch.tensor([1.0, 2, 3, 4, 5]).send(bob)
    p = x.move(alice)

    assert len(alice.object_store._tensors) == 1

    # Test .move on remote objects

    james.clear_objects()
    x = th.tensor([1.0]).send(james)
    remote_x = james.object_store.get_obj(x.id_at_location)
    remote_ptr = remote_x.send(bob)
    assert remote_ptr.id in james.object_store._objects.keys()
    remote_ptr2 = remote_ptr.move(alice)
    assert remote_ptr2.id in james.object_store._objects.keys()

    # Test .move back to myself

    alice.clear_objects()
    bob.clear_objects()
    t = torch.tensor([1.0, 2, 3, 4, 5])
    x = t.send(bob)
    y = x.move(alice)
    z = y.move(me)
    assert (z == t).all()

    # Move object to same location
    alice.clear_objects()
    t = torch.tensor([1.0, 2, 3, 4, 5]).send(bob)
    t = t.move(bob)
    assert torch.all(torch.eq(t.get(), torch.tensor([1.0, 2, 3, 4, 5])))


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


def test_get_remote_shape(workers):
    """Test pointer.shape functionality"""
    bob = workers["bob"]
    # tensor directly sent: shape stored at sending
    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    assert x.shape == torch.Size([5])
    # result of an operation: need to make a call to the remote worker
    y = x + x
    assert y.shape == torch.Size([5])


def test_remote_function_with_multi_ouput(workers):
    """
    Functions like .split return several tensors, registration and response
    must be made carefully in this case
    """
    bob = workers["bob"]

    tensor = torch.tensor([1, 2, 3, 4.0])
    ptr = tensor.send(bob)
    r_ptr = torch.split(ptr, 2)
    assert (r_ptr[0].get() == torch.tensor([1, 2.0])).all()

    tensor = torch.tensor([1, 2, 3, 4.0])
    ptr = tensor.send(bob)
    max_value, argmax_idx = torch.max(ptr, 0)

    assert max_value.get().item() == 4.0
    assert argmax_idx.get().item() == 3


def test_raising_error_when_item_func_called(workers):
    pointer = PointerTensor(id=1000, location=workers["alice"], owner=workers["me"])
    with pytest.raises(RuntimeError):
        pointer.item()


def test_fix_prec_on_pointer_tensor(workers):
    """
    Ensure .fix_precision() works as expected.
    Also check that fix_precision() is not inplace.
    """
    bob = workers["bob"]

    tensor = torch.tensor([1, 2, 3, 4.0])
    ptr = tensor.send(bob)

    ptr_fp = ptr.fix_precision()

    remote_tensor = bob.object_store.get_obj(ptr.id_at_location)
    remote_fp_tensor = bob.object_store.get_obj(ptr_fp.id_at_location)

    # check that fix_precision is not inplace
    assert (remote_tensor == tensor).all()

    assert isinstance(ptr.child, PointerTensor)
    assert isinstance(remote_fp_tensor.child, FixedPrecisionTensor)


def test_fix_prec_on_pointer_of_pointer(workers):
    """
    Ensure .fix_precision() works along a chain of pointers.
    """
    bob = workers["bob"]
    alice = workers["alice"]

    tensor = torch.tensor([1, 2, 3, 4.0])
    ptr = tensor.send(bob)
    ptr = ptr.send(alice)

    ptr = ptr.fix_precision()

    alice_tensor = alice.object_store.get_obj(ptr.id_at_location)
    remote_tensor = bob.object_store.get_obj(alice_tensor.id_at_location)

    assert isinstance(ptr.child, PointerTensor)
    assert isinstance(remote_tensor.child, FixedPrecisionTensor)


def test_float_prec_on_pointer_tensor(workers):
    """
    Ensure .float_precision() works as expected.
    """
    bob = workers["bob"]

    tensor = torch.tensor([1, 2, 3, 4.0])
    ptr = tensor.send(bob)
    ptr = ptr.fix_precision()

    ptr = ptr.float_precision()
    remote_tensor = bob.object_store.get_obj(ptr.id_at_location)

    assert isinstance(ptr.child, PointerTensor)
    assert isinstance(remote_tensor, torch.Tensor)


def test_float_prec_on_pointer_of_pointer(workers):
    """
    Ensure .float_precision() works along a chain of pointers.
    """
    bob = workers["bob"]
    alice = workers["alice"]

    tensor = torch.tensor([1, 2, 3, 4.0])
    ptr = tensor.send(bob)
    ptr = ptr.send(alice)
    ptr = ptr.fix_precision()

    ptr = ptr.float_precision()

    alice_tensor = alice.object_store.get_obj(ptr.id_at_location)
    remote_tensor = bob.object_store.get_obj(alice_tensor.id_at_location)

    assert isinstance(ptr.child, PointerTensor)
    assert isinstance(remote_tensor, torch.Tensor)


def test_share_get(workers):
    """
    Ensure .share() works as expected.
    """
    bob = workers["bob"]
    alice = workers["alice"]
    charlie = workers["charlie"]

    tensor = torch.tensor([1, 2, 3])
    ptr = tensor.send(bob)

    ptr = ptr.share(charlie, alice)
    remote_tensor = bob.object_store.get_obj(ptr.id_at_location)

    assert isinstance(ptr.child, PointerTensor)
    assert isinstance(remote_tensor.child, AdditiveSharingTensor)


def test_registration_of_action_on_pointer_of_pointer(workers):
    """
    Ensure actions along a chain of pointers are registered as expected.
    """
    bob = workers["bob"]
    alice = workers["alice"]

    tensor = torch.tensor([1, 2, 3, 4.0])
    ptr = tensor.send(bob)
    ptr = ptr.send(alice)
    ptr_action = ptr + ptr

    assert len(alice.object_store._tensors) == 2
    assert len(bob.object_store._tensors) == 2


def test_setting_back_grad_to_origin_after_send(workers):
    """
    Calling .backward() on a tensor sent using `.send(..., requires_grad=True)`
    should update the origin tensor gradient
    """
    me = workers["me"]
    alice = workers["alice"]

    with me.registration_enabled():
        x = th.tensor([1.0, 2.0, 3, 4, 5], requires_grad=True)
        y = x + x
        me.register_obj(y)  # registration on the local worker is sometimes buggy

        y_ptr = y.send(alice, requires_grad=True)
        z_ptr = y_ptr * 2

        z = z_ptr.sum()
        z.backward()

        assert (x.grad == th.tensor([4.0, 4.0, 4.0, 4.0, 4.0])).all()


def test_setting_back_grad_to_origin_after_move(workers):
    """
    Calling .backward() on a tensor moved using `.move(..., requires_grad=True)`
    should update the origin tensor gradient
    """
    me = workers["me"]
    bob = workers["bob"]
    alice = workers["alice"]

    with me.registration_enabled():
        x = th.tensor([1.0, 2.0, 3, 4, 5], requires_grad=True)
        y = x + x
        me.register_obj(y)  # registration on the local worker is sometimes buggy

        y_ptr = y.send(alice, requires_grad=True)
        z_ptr = y_ptr * 2

        z_ptr2 = z_ptr.move(bob, requires_grad=True)
        z = z_ptr2.sum()
        z.backward()

        assert (x.grad == th.tensor([4.0, 4.0, 4.0, 4.0, 4.0])).all()


def test_iadd(workers):
    alice = workers["alice"]
    a = torch.ones(1, 5)
    b = torch.ones(1, 5)
    a_pt = a.send(alice)
    b_pt = b.send(alice)

    b_pt += a_pt

    assert len(alice.object_store._objects) == 2


def test_inplace_ops_on_remote_long_tensor(workers):
    alice = workers["alice"]

    t = torch.LongTensor([2])
    p = t.send_(alice) * 2
    p.get_()

    assert p == torch.LongTensor([4])
