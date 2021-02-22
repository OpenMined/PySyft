from json import loads

import pytest
import torch as th
from flask import current_app as app
from syft.core.common import UID
from syft.core.store import Dataset
from syft.core.store.storeable_object import StorableObject
from syft.core.common.serde import _deserialize

from src.main.core.database.store_disk import (
    DiskObjectStore,
    create_dataset,
    create_storable,
)
from src.main.core.database.bin_storage.metadata import StorageMetadata, get_metadata
from src.main.core.database.bin_storage.bin_obj import BinaryObject

storable = create_storable(
    _id=UID(),
    data=th.Tensor([1, 2, 3, 4]),
    description="Dummy tensor",
    tags=["dummy", "tensor"],
)
storable2 = create_storable(
    _id=UID(),
    data=th.Tensor([-1, -2, -3, -4]),
    description="Negative Dummy tensor",
    tags=["negative", "dummy", "tensor"],
)

storable3 = create_storable(
    _id=UID(),
    data=th.Tensor([11, 22, 33, 44]),
    description="NewDummy tensor",
    tags=["new", "dummy", "tensor"],
)

dataset = create_dataset(
    _id=UID(),
    data=[storable, storable2],
    description="Dummy tensor",
    tags=["dummy", "tensor"],
)


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(BinaryObject).delete()
        database.session.query(StorageMetadata).delete()
        database.session.commit()
    except:
        database.session.rollback()


def test_store_item(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    storage.store(dataset)

    assert get_metadata(database).length == 1
    assert database.session.query(BinaryObject).get(dataset.id.value.hex) is not None


def test_store_bytes(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    _id = storage.store_bytes(dataset.to_bytes())

    assert get_metadata(database).length == 1
    assert database.session.query(BinaryObject).get(_id) is not None


def test_contains_true(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    storage.store(dataset)

    assert storage.__contains__(dataset.id.value.hex)


def test_contains_false(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)

    assert not storage.__contains__(dataset.id.value.hex)


def test_setitem(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    storage.store(dataset)
    _id = dataset.id.value.hex

    assert database.session.query(BinaryObject).get(_id) is not None
    old_binary = database.session.query(BinaryObject).get(_id).binary

    new_dataset = create_dataset(
        _id=dataset.id,
        data=[storable2, storable3],
        description="Dummy tensor",
        tags=["dummy", "tensor"],
    )

    storage.__setitem__(dataset.id.value.hex, new_dataset)

    new_binary = database.session.query(BinaryObject).get(_id).binary
    assert hash(new_binary) != hash(old_binary)


def test_delitem(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    bin_obj = BinaryObject(id=dataset.id.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()
    _id = dataset.id.value.hex

    assert database.session.query(BinaryObject).get(_id) is not None
    assert get_metadata(database).length == 1

    storage.__delitem__(_id)

    assert database.session.query(BinaryObject).get(_id) is None
    assert get_metadata(database).length == 0


def test__len__(client, database, cleanup):
    storage = DiskObjectStore(database)

    bin_obj = BinaryObject(id=dataset.id.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    _id = dataset.id.value.hex

    assert storage.__len__() == 1
    obj = database.session.query(BinaryObject).get(dataset.id.value.hex)
    metadata = get_metadata(database)
    metadata.length -= 1

    database.session.delete(obj)
    database.session.commit()

    assert storage.__len__() == 0


def test_get_keys(client, database, cleanup):
    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    bin_obj = BinaryObject(id=uid1.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    keys = storage.keys()
    assert set(keys) == set([uid1.value.hex, uid2.value.hex])


def test_clear(client, database, cleanup):
    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    bin_obj = BinaryObject(id=uid1.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    storage.clear()

    assert database.session.query(BinaryObject).all() == []
    assert database.session.query(StorageMetadata).all() == []


def test__sizeof__(client, database, cleanup):
    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    bin_obj = BinaryObject(id=uid1.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=dataset.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    assert storage.__sizeof__() > 0


def test_get_values(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    new_dataset1 = create_dataset(
        _id=uid1,
        data=[storable2],
        description="Dummy tensor 1",
        tags=["dummy1", "tensor"],
    )

    new_dataset2 = create_dataset(
        _id=uid2,
        data=[storable3],
        description="Dummy tensor 2",
        tags=["dummy2", "tensor"],
    )

    bin_obj = BinaryObject(id=uid1.value.hex, binary=new_dataset1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=new_dataset2.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    binaries = storage.values()

    assert binaries[0].id == new_dataset1.id
    assert binaries[0].description == new_dataset1.description
    assert binaries[0].tags == new_dataset1.tags
    assert th.eq(binaries[0].data[0].data, new_dataset1.data[0].data).all()

    assert binaries[1].id == new_dataset2.id
    assert binaries[1].description == new_dataset2.description
    assert binaries[1].tags == new_dataset2.tags
    assert th.eq(binaries[1].data[0].data, new_dataset2.data[0].data).all()


def test__getitem__(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()

    new_dataset1 = create_dataset(
        _id=uid1,
        data=[storable2],
        description="Dummy tensor 1",
        tags=["dummy1", "tensor"],
    )

    bin_obj = BinaryObject(id=uid1.value.hex, binary=new_dataset1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    retrieved = storage.__getitem__(uid1.value.hex)
    retrieved = _deserialize(blob=retrieved, from_bytes=True)
    assert retrieved.id == new_dataset1.id
    assert retrieved.description == new_dataset1.description
    assert retrieved.tags == new_dataset1.tags
    assert th.eq(retrieved.data[0].data, new_dataset1.data[0].data).all()


def test__getitem__missing(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    new_dataset1 = create_dataset(
        _id=uid1,
        data=[storable2],
        description="Dummy tensor 1",
        tags=["dummy1", "tensor"],
    )

    bin_obj = BinaryObject(id=uid1.value.hex, binary=new_dataset1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    with pytest.raises(AttributeError):
        retrieved = storage.__getitem__(uid2.value.hex)


def test_get_object(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    new_dataset1 = create_dataset(
        _id=uid1,
        data=[storable2],
        description="Dummy tensor 1",
        tags=["dummy1", "tensor"],
    )

    bin_obj = BinaryObject(id=uid1.value.hex, binary=new_dataset1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    retrieved = storage.get_object(uid1.value.hex)
    retrieved = _deserialize(blob=retrieved, from_bytes=True)

    assert retrieved.id == new_dataset1.id
    assert retrieved.description == new_dataset1.description
    assert retrieved.tags == new_dataset1.tags
    assert th.eq(retrieved.data[0].data, new_dataset1.data[0].data).all()


def test_get_object_missing(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    new_dataset1 = create_dataset(
        _id=uid1,
        data=[storable2],
        description="Dummy tensor 1",
        tags=["dummy1", "tensor"],
    )

    bin_obj = BinaryObject(id=uid1.value.hex, binary=new_dataset1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    retrieved = storage.get_object(uid2.value.hex)
    assert retrieved is None
