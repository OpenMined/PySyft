# stdlib
from collections.abc import Callable
from collections.abc import Container
import random
import threading
from typing import Any
from typing import TypeVar

# third party
from faker import Faker
import pytest
from typing_extensions import ParamSpec

# syft absolute
from syft.serde.serializable import serializable
from syft.server.credentials import SyftSigningKey
from syft.server.credentials import SyftVerifyKey
from syft.service.queue.queue_stash import Status
from syft.service.request.request_service import RequestService
from syft.store.db.sqlite import SQLiteDBConfig
from syft.store.db.sqlite import SQLiteDBManager
from syft.store.db.stash import ObjectStash
from syft.store.document_store_errors import NotFoundException
from syft.store.document_store_errors import StashException
from syft.store.linked_obj import LinkedObject
from syft.types.errors import SyftException
from syft.types.syft_object import SyftObject
from syft.types.uid import UID


@serializable()
class MockObject(SyftObject):
    __canonical_name__ = "base_stash_mock_object_type"
    __version__ = 1
    id: UID
    name: str
    desc: str
    importance: int
    value: int
    linked_obj: LinkedObject | None = None
    status: Status = Status.CREATED

    __attr_searchable__ = ["id", "name", "desc", "importance"]
    __attr_unique__ = ["id", "name"]


class MockStash(ObjectStash[MockObject]):
    pass


def get_object_values(obj: SyftObject) -> tuple[Any]:
    return tuple(obj.to_dict().values())


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
def root_verify_key() -> SyftVerifyKey:
    return SyftSigningKey.generate().verify_key


@pytest.fixture
def base_stash(root_verify_key) -> MockStash:
    config = SQLiteDBConfig()
    db_manager = SQLiteDBManager(config, UID(), root_verify_key)
    mock_stash = MockStash(store=db_manager)
    db_manager.init_tables()
    yield mock_stash


def random_sentence(faker: Faker) -> str:
    return faker.paragraph(nb_sentences=1)


def object_kwargs(faker: Faker, **kwargs: Any) -> dict[str, Any]:
    return {
        "name": faker.name(),
        "desc": random_sentence(faker),
        "importance": random.randrange(5),
        "value": random.randrange(100000),
        **kwargs,
    }


def multiple_object_kwargs(
    faker: Faker, n=10, same=False, **kwargs: Any
) -> list[dict[str, Any]]:
    if same:
        kwargs_ = {"id": UID(), **object_kwargs(faker), **kwargs}
        return [kwargs_ for _ in range(n)]
    return [object_kwargs(faker, **kwargs) for _ in range(n)]


@pytest.fixture
def mock_object(faker: Faker) -> MockObject:
    yield MockObject(**object_kwargs(faker))


@pytest.fixture
def mock_objects(faker: Faker) -> list[MockObject]:
    yield [MockObject(**kwargs) for kwargs in multiple_object_kwargs(faker)]


def test_basestash_set(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    result = base_stash.set(root_verify_key, mock_object).unwrap()
    assert result is not None
    assert result == mock_object


def test_basestash_set_duplicate(
    root_verify_key, base_stash: MockStash, faker: Faker
) -> None:
    original, duplicate = (
        MockObject(**kwargs) for kwargs in multiple_object_kwargs(faker, n=2, same=True)
    )

    base_stash.set(root_verify_key, original).unwrap()

    with pytest.raises(StashException):
        base_stash.set(root_verify_key, duplicate).unwrap()


def test_basestash_set_duplicate_unique_key(
    root_verify_key, base_stash: MockStash, faker: Faker
) -> None:
    original, duplicate = (
        MockObject(**kwargs)
        for kwargs in multiple_object_kwargs(faker, n=2, name=faker.name())
    )

    result = base_stash.set(root_verify_key, original)
    assert result.is_ok()

    result = base_stash.set(root_verify_key, duplicate)
    assert result.is_err()


def test_basestash_delete(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    base_stash.set(root_verify_key, mock_object).unwrap()
    base_stash.delete_by_uid(root_verify_key, mock_object.id).unwrap()
    assert len(base_stash.get_all(root_verify_key).unwrap()) == 0


def test_basestash_cannot_delete_non_existent(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    result = base_stash.set(root_verify_key, mock_object).unwrap()

    random_uid = create_unique(UID, [mock_object.id])
    result = base_stash.delete_by_uid(root_verify_key, random_uid)
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
    result = base_stash.set(root_verify_key, mock_object).unwrap()

    updated_obj = mock_object.copy()
    updated_obj.name = faker.name()

    result = base_stash.update(root_verify_key, updated_obj)
    assert result.is_ok()

    retrieved = result.ok()
    assert retrieved == updated_obj


def test_basestash_upsert(
    root_verify_key, base_stash: MockStash, mock_object: MockObject, faker: Faker
) -> None:
    base_stash.set(root_verify_key, mock_object).unwrap()

    updated_obj = mock_object.copy()
    updated_obj.name = faker.name()

    retrieved = base_stash.upsert(root_verify_key, updated_obj).unwrap()
    assert retrieved == updated_obj

    updated_obj.id = UID()

    with pytest.raises(StashException):
        # fails because the name should be unique
        base_stash.upsert(root_verify_key, updated_obj).unwrap()

    updated_obj.name = faker.name()

    retrieved = base_stash.upsert(root_verify_key, updated_obj).unwrap()
    assert retrieved == updated_obj
    assert len(base_stash.get_all(root_verify_key).unwrap()) == 2


def test_basestash_cannot_update_non_existent(
    root_verify_key, base_stash: MockStash, mock_object: MockObject, faker: Faker
) -> None:
    result = base_stash.set(root_verify_key, mock_object).unwrap()

    updated_obj = mock_object.copy()
    updated_obj.id = create_unique(UID, [mock_object.id])
    updated_obj.name = faker.name()

    result = base_stash.update(root_verify_key, updated_obj)
    assert result.is_err()


def test_basestash_set_get_all(
    root_verify_key, base_stash: MockStash, mock_objects: list[MockObject]
) -> None:
    for obj in mock_objects:
        res = base_stash.set(root_verify_key, obj)
        assert res.is_ok()

    stored_objects = base_stash.get_all(
        root_verify_key,
    ).unwrap()
    assert len(stored_objects) == len(mock_objects)

    stored_objects_values = {get_object_values(obj) for obj in stored_objects}
    mock_objects_values = {get_object_values(obj) for obj in mock_objects}
    assert stored_objects_values == mock_objects_values


def test_basestash_get_by_uid(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    result = base_stash.set(root_verify_key, mock_object).unwrap()

    result = base_stash.get_by_uid(root_verify_key, mock_object.id).unwrap()
    assert result == mock_object

    random_uid = create_unique(UID, [mock_object.id])
    bad_uid = base_stash.get_by_uid(root_verify_key, random_uid)
    assert bad_uid.is_err()

    # FIX: Partition should return Ok(None), now it's not consistent. We can get NotFoundException or StashException
    assert (
        isinstance(bad_uid.err(), SyftException)
        or isinstance(bad_uid.err(), StashException)
        or isinstance(bad_uid.err(), NotFoundException)
    )


def test_basestash_delete_by_uid(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    result = base_stash.set(root_verify_key, mock_object).unwrap()

    response = base_stash.delete_by_uid(root_verify_key, mock_object.id).unwrap()
    assert isinstance(response, UID)

    result = base_stash.get_by_uid(root_verify_key, mock_object.id)
    assert result.is_err()

    # FIX: partition None returns are inconsistent; here, we might get NotFoundException or StashException
    assert (
        isinstance(result.err(), SyftException)
        or isinstance(result.err(), StashException)
        or isinstance(result.err(), NotFoundException)
    )


def test_basestash_query_one(
    root_verify_key, base_stash: MockStash, mock_objects: list[MockObject], faker: Faker
) -> None:
    for obj in mock_objects:
        base_stash.set(root_verify_key, obj)

    obj = random.choice(mock_objects)
    result = base_stash.get_one(
        root_verify_key,
        filters={"name": obj.name},
    ).unwrap()

    assert result == obj

    existing_names = {obj.name for obj in mock_objects}
    random_name = create_unique(faker.name, existing_names)

    with pytest.raises(NotFoundException):
        result = base_stash.get_one(
            root_verify_key,
            filters={"name": random_name},
        ).unwrap()

    params = {"name": obj.name, "desc": obj.desc}
    result = base_stash.get_one(
        root_verify_key,
        filters=params,
    ).unwrap()
    assert result == obj

    params = {"name": random_name, "desc": random_sentence(faker)}
    with pytest.raises(NotFoundException):
        result = base_stash.get_one(
            root_verify_key,
            filters=params,
        ).unwrap()


def test_basestash_query_enum(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    base_stash.set(root_verify_key, mock_object).unwrap()
    result = base_stash.get_one(
        root_verify_key,
        filters={"status": Status.CREATED},
    ).unwrap()

    assert result == mock_object
    with pytest.raises(NotFoundException):
        result = base_stash.get_one(
            root_verify_key,
            filters={"status": Status.PROCESSING},
        ).unwrap()


def test_basestash_query_linked_obj(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    mock_object.linked_obj = LinkedObject(
        object_type=MockObject,
        object_uid=UID(),
        id=UID(),
        tags=["tag1", "tag2"],
        server_uid=UID(),
        service_type=RequestService,
    )
    base_stash.set(root_verify_key, mock_object).unwrap()

    result = base_stash.get_one(
        root_verify_key,
        filters={"linked_obj.id": mock_object.linked_obj.id},
    ).unwrap()

    assert result == mock_object


def test_basestash_query_all(
    root_verify_key, base_stash: MockStash, mock_objects: list[MockObject], faker: Faker
) -> None:
    desc = random_sentence(faker)
    n_same = 3
    kwargs_list = multiple_object_kwargs(faker, n=n_same, desc=desc)
    similar_objects = [MockObject(**kwargs) for kwargs in kwargs_list]
    all_objects = mock_objects + similar_objects

    for obj in all_objects:
        base_stash.set(root_verify_key, obj)

    objects = base_stash.get_all(root_verify_key, filters={"desc": desc}).unwrap()
    assert len(objects) == n_same
    assert all(obj.desc == desc for obj in objects)
    original_object_values = {get_object_values(obj) for obj in similar_objects}
    retrived_objects_values = {get_object_values(obj) for obj in objects}
    assert original_object_values == retrived_objects_values

    random_desc = create_unique(
        random_sentence, [obj.desc for obj in all_objects], faker
    )

    objects = base_stash.get_all(
        root_verify_key, filters={"desc": random_desc}
    ).unwrap()
    assert len(objects) == 0

    obj = random.choice(similar_objects)

    params = {"name": obj.name, "desc": obj.desc}
    objects = base_stash.get_all(root_verify_key, filters=params).unwrap()
    assert len(objects) == sum(
        1 for obj_ in all_objects if (obj_.name, obj_.desc) == (obj.name, obj.desc)
    )
    assert objects[0] == obj


def test_basestash_query_all_kwargs_multiple_params(
    root_verify_key, base_stash: MockStash, mock_objects: list[MockObject], faker: Faker
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
    objects = base_stash.get_all(root_verify_key, filters=params).unwrap()
    assert len(objects) == n_same
    assert all(obj.desc == desc for obj in objects)
    original_object_values = {get_object_values(obj) for obj in similar_objects}
    retrived_objects_values = {get_object_values(obj) for obj in objects}
    assert original_object_values == retrived_objects_values

    params = {
        "name": create_unique(faker.name, [obj.name for obj in all_objects]),
        "desc": random_sentence(faker),
    }
    objects = base_stash.get_all(root_verify_key, filters=params).unwrap()
    assert len(objects) == 0

    obj = random.choice(similar_objects)

    params = {"id": obj.id, "name": obj.name, "desc": obj.desc}
    objects = base_stash.get_all(root_verify_key, filters=params).unwrap()
    assert len(objects) == 1
    assert objects[0] == obj


def test_stash_thread_support(
    root_verify_key, base_stash: MockStash, mock_object: MockObject
) -> None:
    assert not base_stash._data
    t = threading.Thread(target=base_stash.set, args=(root_verify_key, mock_object))
    t.start()
    t.join(timeout=5)

    result = base_stash.get_by_uid(root_verify_key, mock_object.id).unwrap()
    assert result == mock_object
