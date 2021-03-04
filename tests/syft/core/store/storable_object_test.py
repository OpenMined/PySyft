# third party
import torch as th

# syft absolute
import syft as sy
from syft import serialize
from syft.core.common import UID
from syft.core.store.storeable_object import StorableObject


def test_create_storable_obj() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    StorableObject(id=id, data=data, description=description, tags=tags)


def test_serde_storable_obj() -> None:
    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    obj = StorableObject(id=id, data=data, description=description, tags=tags)

    blob = sy.serialize(obj=obj)

    sy.deserialize(blob=blob)


def test_serde_storable_obj_2() -> None:
    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    obj = StorableObject(id=id, data=data, description=description, tags=tags)
    blob = serialize(obj)
    ds_obj = sy.deserialize(blob=blob)
    assert obj.id == ds_obj.id
    assert (obj.data == ds_obj.data).all()
    assert obj.description == ds_obj.description
    assert obj.tags == ds_obj.tags


# def test_serde_storable_obj_with_wrapped_class() -> None:
#     """Ensure that storable object serialization works wrapping non-syft classes (like np.ndarray)"""
#
#     id = UID()
#     data = np.array([1, 2, 3, 4])
#     description = "This is a dummy test"
#     tags = ["dummy", "test"]
#     obj = StorableObject(id=id, data=data, description=description, tags=tags)
#
#     blob = sy.serialize(obj=obj)
#
#     sy.deserialize(blob=blob)
