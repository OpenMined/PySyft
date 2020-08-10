import numpy as np

from syft.core.store.storeable_object import StorableObject
from syft.core.store.store_memory import MemoryStore
from syft.core.common import UID


def test_create_memory_storage():
    store = MemoryStore()

    k1 = UID()
    data1 = StorableObject(id=k1, data=np.array([1, 2, 3, 4]), description="This is a dummy test",
                           tags=["dummy", "test"])

    store[k1] = data1
    assert k1 in store
    assert len(store) == 1

    k2 = UID()
    data2 = StorableObject(id=k1, data=np.array([1, 2, 3, 4]), description="This is a dummy test",
                           tags=["dummy", "test"])
    store[k2] = data2

    assert store.get_objects_of_type(np.array) == [data1, data2]
    assert list(store.keys()) == [k1, k2]
    assert list(store.values()) == [data1, data2]

    store.clear()

    assert len(store) == 0
    assert store.__sizeof__() == 48


