# stdlib
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

# syft absolute
from syft.serde.serializable import serializable
from syft.types.syft_object import SYFT_OBJECT_VERSION_1
from syft.types.syft_object import SyftBaseObject
from syft.types.syft_object import SyftHashableObject
from syft.types.syft_object import SyftObject
from syft.types.uid import UID


@serializable(attrs=["key", "value", "flag"])
class MockObject(SyftHashableObject):
    key: str
    value: str
    flag: Optional[bool]

    # Serialize `flag`, but don't use it for hashing
    __hash_exclude_attrs__ = ["flag"]

    def __init__(self, key, value, flag=None):
        self.key = key
        self.value = value
        self.flag = flag


@serializable(attrs=["id", "data"])
class MockWrapper(SyftBaseObject, SyftHashableObject):
    id: str
    data: Optional[MockObject]


@serializable()
class MockSyftObject(SyftObject):
    __canonical_name__ = "MockSyftObject"
    __version__ = SYFT_OBJECT_VERSION_1


@serializable()
class MockNestedList(SyftObject):
    __canonical_name__ = "TestNestedList"
    __version__ = SYFT_OBJECT_VERSION_1

    classes: List[MockSyftObject]


@serializable()
class MockNestedMap(SyftObject):
    __canonical_name__ = "TestNestedMap"
    __version__ = SYFT_OBJECT_VERSION_1

    map: Dict[UID, MockSyftObject]


MOCK_DATA = [
    MockSyftObject(id=UID("99328247805c4bd18cb5dc7388b3bef9")),
    MockSyftObject(id=UID("4e615d65429e4573911346e66cdcd1b6")),
    MockSyftObject(id=UID("8142ae58cd42408691900d9fe5f998bf")),
]


def test_simple_hashing():
    obj1 = MockObject(key="key", value="value")
    obj2 = MockObject(key="key", value="value")

    assert hash(obj1) == hash(obj2)
    assert obj1.hash() == obj2.hash()


def test_nested_hashing():
    common_id = str(uuid4())
    obj1 = MockWrapper(
        id=common_id,
        data=MockObject(key="key", value="value", flag=True),
    )
    obj2 = MockWrapper(
        id=common_id,
        data=MockObject(key="key", value="value", flag=False),
    )

    assert obj1.hash() == obj2.hash()


def test_nested_list():
    obj1 = MockNestedList(
        id=UID("572feb9669204a8cbb41a0ab4cbf0093"),
        classes=deepcopy(MOCK_DATA),
    )
    obj2 = MockNestedList(
        id=UID("572feb9669204a8cbb41a0ab4cbf0093"),
        classes=deepcopy(MOCK_DATA),
    )

    assert obj1.hash() == obj2.hash()


def test_nested_map():
    obj1 = MockNestedMap(
        id=UID("090de5e5f39449a08fde75b0c82ec128"),
        map={d.id: d for d in deepcopy(MOCK_DATA)},
    )
    obj2 = MockNestedMap(
        id=UID("090de5e5f39449a08fde75b0c82ec128"),
        map={d.id: d for d in deepcopy(MOCK_DATA)},
    )

    assert obj1.hash() == obj2.hash()


def test_multithreaded_hashing():
    obj1 = MockNestedMap(
        id=UID("090de5e5f39449a08fde75b0c82ec128"),
        map={d.id: d for d in deepcopy(MOCK_DATA)},
    )
    obj2 = MockNestedMap(
        id=UID("6e9518ea30874948b9049ecb2c1086a9"),
        map={d.id: d for d in deepcopy(MOCK_DATA)},
    )

    def hash_object(obj):
        return obj.hash()

    data = [obj1, obj2, obj1, obj2, obj1, obj2]

    with ThreadPool(4) as pool:
        results = pool.map(hash_object, data, chunksize=1)

    # id = 090de5e5f39449a08fde75b0c82ec128
    assert results[0] == results[2] == results[4]

    # id = 6e9518ea30874948b9049ecb2c1086a9
    assert results[1] == results[3] == results[5]
