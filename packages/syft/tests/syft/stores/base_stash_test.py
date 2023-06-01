# stdlib
import random
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

# third party
from faker import Faker
import pytest
from typing_extensions import ParamSpec

# syft absolute
from syft.serde.serializable import serializable
from syft.store.dict_document_store import DictDocumentStore
from syft.store.document_store import BaseUIDStoreStash
from syft.store.document_store import PartitionKey
from syft.store.document_store import PartitionSettings
from syft.store.document_store import QueryKey
from syft.store.document_store import QueryKeys
from syft.store.document_store import UIDPartitionKey
from syft.types.response import SyftSuccess
from syft.types.syft_object import SyftObject
from syft.types.uid import UID


@serializable()
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


def add_mock_object(root_verify_key, stash: MockStash, obj: MockObject) -> MockObject:
    result = stash.set(root_verify_key, obj)
    assert result.is_ok()

    return result.ok()


T = TypeVar("T")
P = ParamSpec("P")


def create_unique(
    gen: Callable[P, T], xs: Container[T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Generate a value with `gen()` that does not collide with any element in xs"""
    x = gen(*args, **kwargs)
    while x in xs:
        x = gen(*args, **kwargs)

    return x


@pytest.fixture
def base_stash(root_verify_key) -> MockStash:
    return MockStash(store=DictDocumentStore(root_verify_key))


def random_sentence(faker: Faker) -> str:
    return faker.paragraph(nb_sentences=1)


def object_kwargs(faker: Faker, **kwargs: Any) -> Dict[str, Any]:
    return {
        "name": faker.name(),
        "desc": random_sentence(faker),
        "importance": random.randrange(5),
        "value": random.randrange(100000),
        **kwargs,
    }


def multiple_object_kwargs(
    faker: Faker, n=10, same=False, **kwargs: Any
) -> List[Dict[str, Any]]:
    if same:
        kwargs_ = {"id": UID(), **object_kwargs(faker), **kwargs}
        return [kwargs_ for _ in range(n)]
    return [object_kwargs(faker, **kwargs) for _ in range(n)]


@pytest.fixture
def mock_object(faker: Faker) -> MockObject:
    return MockObject(**object_kwargs(faker))


@pytest.fixture
def mock_objects(faker: Faker) -> List[MockObject]:
    return [MockObject(**kwargs) for kwargs in multiple_object_kwargs(faker)]


def test_basestash_set(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    result = add_mock_object(root_verify_key, base_stash, mock_object)

    assert result is not None
    assert result == mock_object


def test_basestash_set_duplicate(
    root_verify_key, base_stash: MockStash, faker: Faker
) -> None:
    original, duplicate = [
        MockObject(**kwargs) for kwargs in multiple_object_kwargs(faker, n=2, same=True)
    ]

    result = base_stash.set(root_verify_key, original)
    assert result.is_ok()

    result = base_stash.set(root_verify_key, duplicate)
    assert result.is_err()


def test_basestash_set_duplicate_unique_key(
    root_verify_key, base_stash: MockStash, faker: Faker
) -> None:
    original, duplicate = [
        MockObject(**kwargs)
        for kwargs in multiple_object_kwargs(faker, n=2, name=faker.name())
    ]

    result = base_stash.set(root_verify_key, original)
    assert result.is_ok()

    result = base_stash.set(root_verify_key, duplicate)
    assert result.is_err()


def test_basestash_delete(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    add_mock_object(root_verify_key, base_stash, mock_object)

    result = base_stash.delete(
        root_verify_key, UIDPartitionKey.with_obj(mock_object.id)
    )
    assert result.is_ok()

    assert len(base_stash.get_all(root_verify_key).ok()) == 0


def test_basestash_cannot_delete_non_existent(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    add_mock_object(root_verify_key, base_stash, mock_object)

    random_uid = create_unique(UID, [mock_object.id])
    for result in [
        base_stash.delete(root_verify_key, UIDPartitionKey.with_obj(random_uid)),
        base_stash.delete_by_uid(root_verify_key, random_uid),
    ]:
        result = base_stash.delete(root_verify_key, UIDPartitionKey.with_obj(UID()))
        assert result.is_err()

    assert (
        len(
            base_stash.get_all(
                root_verify_key,
            ).ok()
        )
        == 1
    )


def test_basestash_update(
    root_verify_key, base_stash: MockStash, mock_object: MockObject, faker: Faker
) -> None:
    add_mock_object(root_verify_key, base_stash, mock_object)

    updated_obj = mock_object.copy()
    updated_obj.name = faker.name()

    result = base_stash.update(root_verify_key, updated_obj)
    assert result.is_ok()

    retrieved = result.ok()
    assert retrieved == updated_obj


def test_basestash_cannot_update_non_existent(
    root_verify_key, base_stash: MockStash, mock_object: MockObject, faker: Faker
) -> None:
    add_mock_object(root_verify_key, base_stash, mock_object)

    updated_obj = mock_object.copy()
    updated_obj.id = create_unique(UID, [mock_object.id])
    updated_obj.name = faker.name()

    result = base_stash.update(root_verify_key, updated_obj)
    assert result.is_err()


def test_basestash_set_get_all(
    root_verify_key, base_stash: MockStash, mock_objects: List[MockObject]
) -> None:
    for obj in mock_objects:
        res = base_stash.set(root_verify_key, obj)
        assert res.is_ok()

    stored_objects = base_stash.get_all(
        root_verify_key,
    )
    assert stored_objects.is_ok()

    stored_objects = stored_objects.ok()
    assert len(stored_objects) == len(mock_objects)

    stored_objects_values = set(get_object_values(obj) for obj in stored_objects)
    mock_objects_values = set(get_object_values(obj) for obj in mock_objects)
    assert stored_objects_values == mock_objects_values


def test_basestash_get_by_uid(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    add_mock_object(root_verify_key, base_stash, mock_object)

    result = base_stash.get_by_uid(root_verify_key, mock_object.id)
    assert result.is_ok()
    assert result.ok() == mock_object

    random_uid = create_unique(UID, [mock_object.id])
    result = base_stash.get_by_uid(root_verify_key, random_uid)
    assert result.is_ok()
    assert result.ok() is None


def test_basestash_delete_by_uid(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    add_mock_object(root_verify_key, base_stash, mock_object)

    result = base_stash.delete_by_uid(root_verify_key, mock_object.id)
    assert result.is_ok()
    response = result.ok()
    assert isinstance(response, SyftSuccess)

    result = base_stash.get_by_uid(root_verify_key, mock_object.id)
    assert result.is_ok()
    assert result.ok() is None


def test_basestash_query_one(
    root_verify_key, base_stash: MockStash, mock_objects: List[MockObject], faker: Faker
) -> None:
    for obj in mock_objects:
        base_stash.set(root_verify_key, obj)

    obj = random.choice(mock_objects)

    for result in (
        base_stash.query_one_kwargs(root_verify_key, name=obj.name),
        base_stash.query_one(
            root_verify_key, QueryKey.from_obj(NamePartitionKey, obj.name)
        ),
    ):
        assert result.is_ok()
        assert result.ok() == obj

    existing_names = set(obj.name for obj in mock_objects)
    random_name = create_unique(faker.name, existing_names)

    for result in (
        base_stash.query_one_kwargs(root_verify_key, name=random_name),
        base_stash.query_one(
            root_verify_key, QueryKey.from_obj(NamePartitionKey, random_name)
        ),
    ):
        assert result.is_ok()
        assert result.ok() is None

    params = {"name": obj.name, "desc": obj.desc}
    for result in [
        base_stash.query_one_kwargs(root_verify_key, **params),
        base_stash.query_one(root_verify_key, QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        assert result.ok() == obj

    params = {"name": random_name, "desc": random_sentence(faker)}
    for result in [
        base_stash.query_one_kwargs(root_verify_key, **params),
        base_stash.query_one(root_verify_key, QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        assert result.ok() is None


def test_basestash_query_all(
    root_verify_key, base_stash: MockStash, mock_objects: List[MockObject], faker: Faker
) -> None:
    desc = random_sentence(faker)
    n_same = 3
    kwargs_list = multiple_object_kwargs(faker, n=n_same, desc=desc)
    similar_objects = [MockObject(**kwargs) for kwargs in kwargs_list]
    all_objects = mock_objects + similar_objects

    for obj in all_objects:
        base_stash.set(root_verify_key, obj)

    for result in [
        base_stash.query_all_kwargs(root_verify_key, desc=desc),
        base_stash.query_all(
            root_verify_key, QueryKey.from_obj(DescPartitionKey, desc)
        ),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == n_same
        assert all(obj.desc == desc for obj in objects)
        original_object_values = set(get_object_values(obj) for obj in similar_objects)
        retrived_objects_values = set(get_object_values(obj) for obj in objects)
        assert original_object_values == retrived_objects_values

    random_desc = create_unique(
        random_sentence, [obj.desc for obj in all_objects], faker
    )
    for result in [
        base_stash.query_all_kwargs(root_verify_key, desc=random_desc),
        base_stash.query_all(
            root_verify_key, QueryKey.from_obj(DescPartitionKey, random_desc)
        ),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == 0

    obj = random.choice(similar_objects)

    params = {"name": obj.name, "desc": obj.desc}
    for result in [
        base_stash.query_all_kwargs(root_verify_key, **params),
        base_stash.query_all(root_verify_key, QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == sum(
            1 for obj_ in all_objects if (obj_.name, obj_.desc) == (obj.name, obj.desc)
        )
        assert objects[0] == obj


def test_basestash_query_all_kwargs_multiple_params(
    root_verify_key, base_stash: MockStash, mock_objects: List[MockObject], faker: Faker
) -> None:
    desc = random_sentence(faker)
    importance = random.randrange(5)
    n_same = 3
    kwargs_list = multiple_object_kwargs(
        faker, n=n_same, importance=importance, desc=desc
    )
    similar_objects = [MockObject(**kwargs) for kwargs in kwargs_list]
    all_objects = mock_objects + similar_objects

    for obj in all_objects:
        base_stash.set(root_verify_key, obj)

    params = {"importance": importance, "desc": desc}
    for result in [
        base_stash.query_all_kwargs(root_verify_key, **params),
        base_stash.query_all(root_verify_key, QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == n_same
        assert all(obj.desc == desc for obj in objects)
        original_object_values = set(get_object_values(obj) for obj in similar_objects)
        retrived_objects_values = set(get_object_values(obj) for obj in objects)
        assert original_object_values == retrived_objects_values

    params = {
        "name": create_unique(faker.name, [obj.name for obj in all_objects]),
        "desc": random_sentence(faker),
    }
    for result in [
        base_stash.query_all_kwargs(root_verify_key, **params),
        base_stash.query_all(root_verify_key, QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == 0

    obj = random.choice(similar_objects)

    params = {"id": obj.id, "name": obj.name, "desc": obj.desc}
    for result in [
        base_stash.query_all_kwargs(root_verify_key, **params),
        base_stash.query_all(root_verify_key, QueryKeys.from_dict(params)),
    ]:
        assert result.is_ok()
        objects = result.ok()
        assert len(objects) == 1
        assert objects[0] == obj


def test_basestash_cannot_query_non_searchable(
    root_verify_key, base_stash: MockStash, mock_objects: List[MockObject]
) -> None:
    for obj in mock_objects:
        base_stash.set(root_verify_key, obj)

    obj = random.choice(mock_objects)

    assert base_stash.query_one_kwargs(root_verify_key, value=10).is_err()
    assert base_stash.query_all_kwargs(root_verify_key, value=10).is_err()
    assert base_stash.query_one_kwargs(
        root_verify_key, value=10, name=obj.name
    ).is_err()
    assert base_stash.query_all_kwargs(
        root_verify_key, value=10, name=obj.name
    ).is_err()

    ValuePartitionKey = PartitionKey(key="value", type_=int)
    qk = ValuePartitionKey.with_obj(10)

    assert base_stash.query_one(root_verify_key, qk).is_err()
    assert base_stash.query_all(root_verify_key, qk).is_err()
    assert base_stash.query_all(root_verify_key, QueryKeys(qks=[qk])).is_err()
    assert base_stash.query_all(
        root_verify_key, QueryKeys(qks=[qk, UIDPartitionKey.with_obj(obj.id)])
    ).is_err()
