# syft absolute
import syft as sy
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
    data = UID()
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    obj = StorableObject(id=id, data=data, description=description, tags=tags)

    blob = sy.serialize(obj=obj)

    sy.deserialize(blob=blob)


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
