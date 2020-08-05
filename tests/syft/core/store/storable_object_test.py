import syft as sy
import numpy as np

from syft.core.store import StorableObject
from syft.core.common import UID


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

def test_serde_storable_obj_with_wrapped_class():
    """Ensure that storable object serialization works wrapping non-syft classes (like np.ndarray)"""


    key = UID()
    data = np.array([1,2,3,4])
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    obj = StorableObject(key=key, data=data, description=description, tags=tags)

    blob = sy.serialize(obj=obj)

    sy.deserialize(blob=blob)
