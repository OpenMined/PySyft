import torch as th
import re

from syft.core.store.storeable_object import StorableObject
from syft.core.store.store_memory import MemoryStore
from syft.core.common import UID

from typing import Tuple, List


def generate_id_obj(data: th.Tensor, description: str, tags: List[str]) -> \
        Tuple[UID, StorableObject]:
    id = UID()
    obj = StorableObject(
        id=id,
        data=data,
        description=description,
        tags=tags
    )

    return id, obj


def test_create_memory_storage() -> None:
    store = MemoryStore()


def test_set_item() -> None:
    store = MemoryStore()
    id1, obj1 = generate_id_obj(data=th.Tensor([1, 2, 3, 4]),
                                description="Dummy tensor",
                                tags=["dummy", "tensor"])

    store[id1] = obj1
    assert id1 in store
    assert len(store) == 1


def test_store_item() -> None:
    store = MemoryStore()
    id1, obj1 = generate_id_obj(data=th.Tensor([1, 2, 3, 4]),
                                description="Dummy tensor",
                                tags=["dummy", "tensor"])

    store.store(obj=obj1)
    assert id1 in store
    assert len(store) == 1


def test_get_objects_of_type() -> None:
    store = MemoryStore()
    id1, obj1 = generate_id_obj(data=th.Tensor([1, 2, 3, 4]),
                                description="Dummy tensor",
                                tags=["dummy", "tensor"])
    id2, obj2 = generate_id_obj(data=th.Tensor([1, 2, 3]),
                                description="Another dummy tensor",
                                tags=["another", "dummy", "tensor"])

    store[id1] = obj1
    store[id2] = obj2
    assert store.get_objects_of_type(th.Tensor) == [obj1, obj2]


def test_keys_values() -> None:
    store = MemoryStore()
    id1, obj1 = generate_id_obj(data=th.Tensor([1, 2, 3, 4]),
                                description="Dummy tensor",
                                tags=["dummy", "tensor"])
    id2, obj2 = generate_id_obj(data=th.Tensor([1, 2, 3]),
                                description="Another dummy tensor",
                                tags=["another", "dummy", "tensor"])

    store[id1] = obj1
    store[id2] = obj2
    assert list(store.keys()) == [id1, id2]
    assert list(store.values()) == [obj1, obj2]


def test_clear_len() -> None:
    store = MemoryStore()
    id1, obj1 = generate_id_obj(data=th.Tensor([1, 2, 3, 4]),
                                description="Dummy tensor",
                                tags=["dummy", "tensor"])
    id2, obj2 = generate_id_obj(data=th.Tensor([1, 2, 3]),
                                description="Another dummy tensor",
                                tags=["another", "dummy", "tensor"])

    store[id1] = obj1
    store[id2] = obj2

    assert len(store) == 2
    store.clear()
    assert len(store) == 0


def test_str() -> None:
    store = MemoryStore()
    id1, obj1 = generate_id_obj(data=th.Tensor([1, 2, 3, 4]),
                                description="Dummy tensor",
                                tags=["dummy", "tensor"])

    store[id1] = obj1
    store_regex = re.compile(
        r'^{<UID:[0-9a-z]{8}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{12}>:'
        r' <Storable:tensor\(\[1\., 2\., 3\., 4\.\]\)>}$')
    assert store_regex.match(str(store))
