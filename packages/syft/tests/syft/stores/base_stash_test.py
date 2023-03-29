# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# third party
import pytest

# syft absolute
from syft.core.node.new.dict_document_store import DictDocumentStore
from syft.core.node.new.document_store import BaseUIDStoreStash
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import UIDPartitionKey
from syft.core.node.new.response import SyftSuccess
from syft.core.node.new.serializable import serializable
from syft.core.node.new.syft_object import SyftObject
from syft.core.node.new.uid import UID


@serializable(recursive_serde=True)
class MockObject(SyftObject):
    __canonical_name__ = "base_stash_mock_object_type"
    id: UID
    name: str

    __attr_searchable__ = ["id", "name"]
    __attr_unique__ = ["id"]


class MockStash(BaseUIDStoreStash):
    object_type = MockObject
    settings = PartitionSettings(
        name=MockObject.__canonical_name__, object_type=MockObject
    )


def get_object_values(obj: SyftObject) -> Tuple[Any]:
    return tuple(obj.dict().values())


def add_mock_object(stash: MockStash, obj: MockObject) -> MockObject:
    result = stash.set(obj)
    assert result.is_ok()

    return result.ok()


@pytest.fixture
def base_stash() -> MockStash:
    return MockStash(store=DictDocumentStore())


def object_kwargs(faker) -> Dict[str, Any]:
    return {"name": faker.name()}


def multiple_object_kwargs(faker, n=10, same=False) -> List[Dict[str, Any]]:
    if same:
        kwargs = {"id": UID(), **object_kwargs(faker)}
        return [kwargs for _ in range(n)]
    return [object_kwargs(faker) for _ in range(n)]


@pytest.fixture
def mock_object(faker) -> MockObject:
    return MockObject(**object_kwargs(faker))


@pytest.fixture
def mock_objects(faker) -> List[MockObject]:
    return [MockObject(**kwargs) for kwargs in multiple_object_kwargs(faker)]


def test_basestash_set(base_stash: MockStash, mock_object: MockObject) -> None:
    result = add_mock_object(base_stash, mock_object)

    assert result is not None
    assert result == mock_object


def test_basestash_set_duplicate(base_stash: MockStash, faker) -> None:
    original, duplicate = [
        MockObject(**kwargs) for kwargs in multiple_object_kwargs(faker, n=2, same=True)
    ]

    result = base_stash.set(original)
    assert result.is_ok()

    result = base_stash.set(duplicate)
    assert result.is_err()


def test_basestash_delete(base_stash: MockStash, mock_object: MockObject) -> None:
    add_mock_object(base_stash, mock_object)

    result = base_stash.delete(UIDPartitionKey.with_obj(mock_object.id))
    assert result.is_ok()

    assert len(base_stash.get_all().ok()) == 0


def test_basestash_update(
    base_stash: MockStash, mock_object: MockObject, faker
) -> None:
    add_mock_object(base_stash, mock_object)

    updated_obj = mock_object.copy()
    updated_obj.name = faker.name()

    result = base_stash.update(updated_obj)
    assert result.is_ok()

    retrieved = result.ok()
    assert retrieved == updated_obj


def test_basestash_set_get_all(
    base_stash: MockStash, mock_objects: List[MockObject]
) -> None:
    for obj in mock_objects:
        base_stash.set(obj)

    stored_objects = base_stash.get_all()
    assert stored_objects.is_ok()

    stored_objects = stored_objects.ok()
    assert len(stored_objects) == len(mock_objects)

    stored_objects_values = set(get_object_values(obj) for obj in stored_objects)
    mock_objects_values = set(get_object_values(obj) for obj in mock_objects)
    assert stored_objects_values == mock_objects_values


def test_basestash_get_by_uid(base_stash: MockStash, mock_object: MockObject) -> None:
    add_mock_object(base_stash, mock_object)

    result = base_stash.get_by_uid(mock_object.id)
    assert result.is_ok()
    assert result.ok() == mock_object

    random_uid = UID()
    result = base_stash.get_by_uid(random_uid)
    assert result.is_ok()
    assert result.ok() is None


def test_basestash_delete_by_uid(
    base_stash: MockStash, mock_object: MockObject
) -> None:
    add_mock_object(base_stash, mock_object)

    result = base_stash.delete_by_uid(mock_object.id)
    assert result.is_ok()
    response = result.ok()
    assert isinstance(response, SyftSuccess)

    result = base_stash.get_by_uid(mock_object.id)
    assert result.is_ok()
    assert result.ok() is None
