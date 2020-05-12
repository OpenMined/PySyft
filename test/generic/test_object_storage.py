import torch

from syft.generic import object_storage


def test_clear_objects():
    """
    Checks the clear_objects method
    """
    obj_storage = (
        object_storage.ObjectStore()
    )  #  obj_storage is a wrapper object to a collection of objects

    x = torch.tensor(1)
    obj_storage.set_obj(x)

    objs = obj_storage.current_objects()  # Returns a copy of the objects in obj_storage(here:x)

    assert len(objs) == 1
    assert objs[x.id] == x

    ret_val = obj_storage.clear_objects()  # Completely removes all objects from obj_storage

    objs = obj_storage.current_objects()
    assert len(objs) == 0
    assert ret_val is None


def test_set_obj_takes_ownership(workers):
    me = workers["me"]
    bob = workers["bob"]

    x = torch.tensor(1)

    x.owner = bob

    me.set_obj(x)

    objs = me.object_store._objects

    assert objs[x.id] == x
    assert objs[x.id].owner == workers["me"]
