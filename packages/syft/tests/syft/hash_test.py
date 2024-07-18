# stdlib
from uuid import uuid4

# syft absolute
from syft.serde.serializable import serializable
from syft.types.syft_object import SYFT_OBJECT_VERSION_1
from syft.types.syft_object import SyftBaseObject
from syft.types.syft_object import SyftHashableObject


@serializable(
    attrs=["key", "value", "flag"],
    canonical_name="MockObject",
    version=1,
)
class MockObject(SyftHashableObject):
    key: str
    value: str
    flag: bool | None

    # Serialize `flag`, but don't use it for hashing
    __hash_exclude_attrs__ = ["flag"]

    def __init__(self, key, value, flag=None):
        self.key = key
        self.value = value
        self.flag = flag


@serializable(attrs=["id", "data"])
class MockWrapper(SyftBaseObject, SyftHashableObject):
    __canonical_name__ = "MockWrapper"
    __version__ = SYFT_OBJECT_VERSION_1

    id: str
    data: MockObject | None


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
