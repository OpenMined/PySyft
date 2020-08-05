import syft as sy
from syft.core.store import StorableObject
from syft.core.common import UID
from syft.core.common.serde import _deserialize


def test_create_storable_obj():
    key = UID()
    data = UID()
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    StorableObject(key=key, data=data, description=description, tags=tags)


def test_serde_storable_obj():
    key = UID()
    data = UID()
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    obj = StorableObject(key=key, data=data, description=description, tags=tags)

    blob = sy.serialize(obj=obj)

    sy.deserialize(blob=blob)
