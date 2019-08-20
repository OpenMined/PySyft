import torch
from syft.generic.pointers import callable_pointer
from syft.generic import pointers


def test_create_callable_pointer(workers):
    alice = workers["alice"]
    bob = workers["bob"]
    callable_pointer.create_callable_pointer(
        id=500,
        id_at_location=2,
        location=alice,
        owner=bob,
        tags="tags",
        description="description",
        register_pointer=True,
    )

    assert len(alice._objects) == 0
    assert len(bob._objects) == 1

    callable_pointer.create_callable_pointer(
        id=501,
        id_at_location=2,
        location=alice,
        owner=bob,
        tags="tags",
        description="description",
        register_pointer=False,
    )

    assert len(alice._objects) == 0
    assert len(bob._objects) == 1


def test_get_obj_callable_pointer(workers):
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

    assert len(alice._objects) == 1
    assert len(bob._objects) == 1

    x_get = obj_ptr.get()

    assert len(alice._objects) == 0
    assert len(bob._objects) == 0
    assert x_get == x


def test_call_callable_pointer(workers):
    def foo(x):
        return x + 2

    alice = workers["alice"]
    bob = workers["bob"]

    id_alice = 100
    id_bob = 200
    foo_wrapper = pointers.ObjectWrapper(id=id_alice, obj=foo)

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
