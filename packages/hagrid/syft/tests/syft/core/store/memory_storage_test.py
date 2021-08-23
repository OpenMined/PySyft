"""In this test suite, we evaluate the MemoryStore class. For more info
on the MemoryStore class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for ways MemoryStore can be initialized
    - CLASS METHODS: tests for the use of MemoryStore's class methods
"""

# stdlib
import re
from typing import List
from typing import Tuple

# third party
import torch as th

# syft absolute
from syft.core.common import UID
from syft.core.common.object import ObjectWithID
from syft.core.store import ObjectStore
from syft.core.store.store_memory import MemoryStore
from syft.core.store.storeable_object import StorableObject


def generate_id_obj(
    data: th.Tensor, description: str, tags: List[str]
) -> Tuple[UID, StorableObject]:
    id = UID()
    obj = StorableObject(id=id, data=data, description=description, tags=tags)

    return id, obj


# --------------------- INITIALIZATION ---------------------


def test_create_memory_storage() -> None:
    """Test that creating MemoryStore() does in fact create
    an ObjectStore."""

    store = MemoryStore()
    assert isinstance(store, ObjectStore)


# --------------------- CLASS METHODS ---------------------


def test_set_item() -> None:
    """Tests that __setitem__ and __getitem__ work intuitively."""

    store = MemoryStore()
    id1, obj1 = generate_id_obj(
        data=th.Tensor([1, 2, 3, 4]),
        description="Dummy tensor",
        tags=["dummy", "tensor"],
    )

    store[id1] = obj1
    assert id1 in store
    assert len(store) == 1


def test_store_item() -> None:
    """Tests that store() works as an alternative to __setitem__."""

    store = MemoryStore()
    id1, obj1 = generate_id_obj(
        data=th.Tensor([1, 2, 3, 4]),
        description="Dummy tensor",
        tags=["dummy", "tensor"],
    )

    store[id1] = obj1
    assert id1 in store
    assert len(store) == 1


def test_get_objects_of_type() -> None:
    """Tests that get_objects_of_type() allows MemoryStore to filter data
    types."""

    store = MemoryStore()
    id1, obj1 = generate_id_obj(
        data=th.Tensor([1, 2, 3, 4]),
        description="Dummy tensor",
        tags=["dummy", "tensor"],
    )
    id2, obj2 = generate_id_obj(
        data=th.Tensor([1, 2, 3]),
        description="Another dummy tensor",
        tags=["another", "dummy", "tensor"],
    )
    id3, obj3 = generate_id_obj(
        data=ObjectWithID(),
        description="Dummy object with ID",
        tags=["dummy", "object", "with", "id"],
    )

    store[id1] = obj1
    store[id2] = obj2
    store[id3] = obj3
    assert all(x in store.get_objects_of_type(th.Tensor) for x in [obj1, obj2])
    assert obj3 in store.get_objects_of_type(ObjectWithID)


def test_keys_values() -> None:
    """Tests that keys() and values() work intuitively and offer MemoryStore
    a dict-like usage."""

    store = MemoryStore()
    id1, obj1 = generate_id_obj(
        data=th.Tensor([1, 2, 3, 4]),
        description="Dummy tensor",
        tags=["dummy", "tensor"],
    )
    id2, obj2 = generate_id_obj(
        data=th.Tensor([1, 2, 3]),
        description="Another dummy tensor",
        tags=["another", "dummy", "tensor"],
    )

    store[id1] = obj1
    store[id2] = obj2
    assert list(store.keys()) == [id1, id2]
    assert list(store.values()) == [obj1, obj2]


def test_clear_len() -> None:
    """Tests that clear() empties the MemoryStore and len() returns the
    number of stored objects."""

    store = MemoryStore()
    id1, obj1 = generate_id_obj(
        data=th.Tensor([1, 2, 3, 4]),
        description="Dummy tensor",
        tags=["dummy", "tensor"],
    )
    id2, obj2 = generate_id_obj(
        data=th.Tensor([1, 2, 3]),
        description="Another dummy tensor",
        tags=["another", "dummy", "tensor"],
    )

    store[id1] = obj1
    store[id2] = obj2

    assert len(store) == 2
    store.clear()
    assert len(store) == 0


def test_str() -> None:
    """Tests that MemoryStore is converted to string properly."""

    store = MemoryStore()
    id1, obj1 = generate_id_obj(
        data=th.Tensor([1, 2, 3, 4]),
        description="Dummy tensor",
        tags=["dummy", "tensor"],
    )

    store[id1] = obj1
    store_regex = re.compile(
        r"^OrderedDict\(\[\(<UID: [0-9a-z]{8}[0-9a-z]{4}[0-9a-z]{4}[0-9a-z]{4}[0-9a-z]"
        + r"{12}>, <Storable: tensor\(\[1\., 2\., 3\., 4\.\]\)>\)\]\)$"
    )
    assert store_regex.match(str(store))
