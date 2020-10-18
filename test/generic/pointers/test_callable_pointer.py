import torch
from syft.generic.pointers import callable_pointer
from syft.generic.pointers.object_wrapper import ObjectWrapper


def test_create_callable_pointer(workers):
    """
    Asserts that a callable pointer is correctly created.
    """
    alice = workers["alice"]
    bob = workers["bob"]
    p = callable_pointer.create_callable_pointer(
        id=500,
        id_at_location=2,
        location=alice,
        owner=bob,
        tags="tags",
        description="description",
        register_pointer=True,
    )

    assert len(alice.object_store._tensors) == 0
    assert isinstance(bob.object_store.get_obj(500), callable_pointer.CallablePointer)

    p = callable_pointer.create_callable_pointer(
        id=501,
        id_at_location=2,
        location=alice,
        owner=bob,
        tags="tags",
        description="description",
        register_pointer=False,
    )

    assert len(alice.object_store._tensors) == 0
    assert isinstance(bob.object_store.get_obj(500), callable_pointer.CallablePointer)
    assert 501 not in bob.object_store._objects


def test_get_obj_callable_pointer(workers):
    """
    Asserts that correct object values are returned when
    `callable_pointer` is called.
    """
    alice = workers["alice"]
    bob = workers["bob"]

    x = torch.tensor(5)
    x_ptr = x.send(alice)

    obj_ptr = callable_pointer.create_callable_pointer(
        id=1,
        id_at_location=x_ptr.id_at_location,
        location=alice,
        owner=bob,
        tags="tags",
        description="description",
        register_pointer=True,
    )

    assert len(alice.object_store._tensors) == 1
    assert isinstance(bob.object_store.get_obj(1), callable_pointer.CallablePointer)

    x_get = obj_ptr.get()

    assert len(alice.object_store._tensors) == 0
    assert len(bob.object_store._tensors) == 0
    assert 1 not in bob.object_store._objects
    assert x_get == x


def test_call_callable_pointer(workers):
    """
    Tests that the correct result after an operation is
    returned when `callable_pointer` is called.
    """

    def foo(x):
        """ Adds 2 to a given input `x`."""
        return x + 2

    alice = workers["alice"]
    bob = workers["bob"]

    id_alice = 100
    id_bob = 200
    foo_wrapper = ObjectWrapper(id=id_alice, obj=foo)

    alice.register_obj(foo_wrapper, id_alice)

    foo_ptr = callable_pointer.create_callable_pointer(
        id=id_bob,
        id_at_location=id_alice,
        location=alice,
        owner=bob,
        tags="tags",
        description="description",
        register_pointer=True,
    )

    res = foo_ptr(4)

    assert res == 6
