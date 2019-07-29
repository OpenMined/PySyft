import torch

from syft.generic import object_storage


def test_clear_objects():
    obj_storage = object_storage.ObjectStorage()

    x = torch.tensor(1)
    obj_storage.set_obj(x)

    objs = obj_storage.current_objects()

    assert len(objs) == 1
    assert objs[x.id] == x

    obj_storage.clear_objects()

    objs = obj_storage.current_objects()
    assert len(objs) == 0
