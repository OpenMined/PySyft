# stdlib
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# third party
import pytest

# syft absolute
from syft.core.node.new.dict_document_store import DictDocumentStore
from syft.core.node.new.document_store import BaseUIDStoreStash
from syft.core.node.new.document_store import PartitionKey
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import QueryKey
from syft.core.node.new.document_store import QueryKeys
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
    desc: str
    importance: int
    value: int

    __attr_searchable__ = ["id", "name", "desc", "importance"]
    __attr_unique__ = ["id", "name"]


NamePartitionKey = PartitionKey(key="name", type_=str)
DescPartitionKey = PartitionKey(key="desc", type_=str)
ImportancePartitionKey = PartitionKey(key="importance", type_=int)


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


def random_sentence(faker):
    return faker.paragraph(nb_sentences=1)


def object_kwargs(faker, **kwargs) -> Dict[str, Any]:
    return {
        "name": faker.name(),
        "desc": random_sentence(faker),
        "importance": random.randrange(5),
        "value": random.randrange(100000),
        **kwargs,
    }


def multiple_object_kwargs(faker, n=10, same=False, **kwargs) -> List[Dict[str, Any]]:
    if same:
        kwargs_ = {"id": UID(), **object_kwargs(faker), **kwargs}
        return [kwargs_ for _ in range(n)]
    return [object_kwargs(faker, **kwargs) for _ in range(n)]


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


def test_basestash_set_duplicate_unique_key(base_stash: MockStash, faker) -> None:
    original, duplicate = [
        MockObject(**kwargs)
        for kwargs in multiple_object_kwargs(faker, n=2, name=faker.name())
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


def test_basestash_query_one(
    base_stash: MockStash, mock_objects: List[MockObject], faker
) -> None:
    for obj in mock_objects:
        base_stash.set(obj)

    obj = random.choice(mock_objects)

    for result in (
        base_stash.query_one_kwargs(name=obj.name),
        base_stash.query_one(QueryKey.from_obj(NamePartitionKey, obj.name)),
    ):
        assert result.is_ok()
        assert result.ok() == obj

    random_name = faker.name()
    for result in (
        base_stash.query_one_kwargs(name=random_name),
        base_stash.query_one(QueryKey.from_obj(NamePartitionKey, random_name)),
    ):
        assert result.is_ok()
        assert result.ok() is None

    params = {"name": obj.name, "desc": obj.desc}
    for result in [
        base_stash.query_one_kwargs(**params),
        base_stash.query_one(QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        assert result.ok() == obj

    params = {"name": faker.name(), "desc": random_sentence(faker)}
    for result in [
        base_stash.query_one_kwargs(**params),
        base_stash.query_one(QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        assert result.ok() is None


def test_basestash_query_all(
    base_stash: MockStash, mock_objects: List[MockObject], faker
) -> None:
    desc = random_sentence(faker)
    n_same = 3
    kwargs_list = multiple_object_kwargs(faker, n=n_same, desc=desc)
    similar_objects = [MockObject(**kwargs) for kwargs in kwargs_list]

    for obj in mock_objects + similar_objects:
        base_stash.set(obj)

    for result in [
        base_stash.query_all_kwargs(desc=desc),
        base_stash.query_all(QueryKey.from_obj(DescPartitionKey, desc)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == n_same
        assert all(obj.desc == desc for obj in objects)
        original_object_values = set(get_object_values(obj) for obj in similar_objects)
        retrived_objects_values = set(get_object_values(obj) for obj in objects)
        assert original_object_values == retrived_objects_values

    random_desc = random_sentence(faker)
    for result in [
        base_stash.query_all_kwargs(desc=random_desc),
        base_stash.query_all(QueryKey.from_obj(DescPartitionKey, random_desc)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == 0

    obj = random.choice(similar_objects)

    params = {"name": obj.name, "desc": obj.desc}
    for result in [
        base_stash.query_all_kwargs(**params),
        base_stash.query_all(QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == 1
        assert objects[0] == obj


def test_basestash_query_all_kwargs_multiple_params(
    base_stash: MockStash, mock_objects: List[MockObject], faker
) -> None:
    desc = random_sentence(faker)
    importance = random.randrange(5)
    n_same = 3
    kwargs_list = multiple_object_kwargs(
        faker, n=n_same, importance=importance, desc=desc
    )
    similar_objects = [MockObject(**kwargs) for kwargs in kwargs_list]

    for obj in mock_objects + similar_objects:
        base_stash.set(obj)

    params = {"importance": importance, "desc": desc}
    for result in [
        base_stash.query_all_kwargs(**params),
        base_stash.query_all(QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == n_same
        assert all(obj.desc == desc for obj in objects)
        original_object_values = set(get_object_values(obj) for obj in similar_objects)
        retrived_objects_values = set(get_object_values(obj) for obj in objects)
        assert original_object_values == retrived_objects_values

    params = {"name": faker.name(), "desc": random_sentence(faker)}
    for result in [
        base_stash.query_all_kwargs(**params),
        base_stash.query_all(QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == 0

    obj = random.choice(similar_objects)

    params = {"id": obj.id, "name": obj.name, "desc": obj.desc}
    for result in [
        base_stash.query_all_kwargs(**params),
        base_stash.query_all(QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == 1
        assert objects[0] == obj


def test_basestash_cannot_query_non_searchable(
    base_stash: MockStash, mock_objects: List[MockObject], faker
) -> None:
    for obj in mock_objects:
        base_stash.set(obj)

    obj = random.choice(mock_objects)

    assert base_stash.query_one_kwargs(value=10).is_err()
    assert base_stash.query_all_kwargs(value=10).is_err()
    assert base_stash.query_one_kwargs(value=10, name=obj.name).is_err()
    assert base_stash.query_all_kwargs(value=10, name=obj.name).is_err()

    ValuePartitionKey = PartitionKey(key="value", type_=int)
    qk = ValuePartitionKey.with_obj(10)

    assert base_stash.query_one(qk).is_err()
    assert base_stash.query_all(qk).is_err()
    assert base_stash.query_all(QueryKeys(qks=[qk])).is_err()
    assert base_stash.query_all(
        QueryKeys(qks=[qk, UIDPartitionKey.with_obj(obj.id)])
    ).is_err()
