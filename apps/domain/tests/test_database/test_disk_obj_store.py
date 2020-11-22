from json import loads

import pytest
import torch as th
from flask import current_app as app
from syft.core.common import UID
from syft.core.store.storeable_object import StorableObject

from src.main.core.database.store_disk import DiskObjectStore, create_storable
from src.main.core.database.bin_storage.metadata import StorageMetadata, get_metadata
from src.main.core.database.bin_storage.bin_obj import BinaryObject

storable = create_storable(
    _id=UID(),
    data=th.Tensor([1, 2, 3, 4]),
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
    storage.store(storable)

    assert get_metadata(database).length == 1
    assert database.session.query(BinaryObject).get(storable.id.value.hex) is not None


def test_contains_true(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    storage.store(storable)

    assert storage.__contains__(storable.id)


def test_contains_false(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)

    assert not storage.__contains__(storable.id)


def test_setitem(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    storage.store(storable)
    _id = storable.id.value.hex

    assert database.session.query(BinaryObject).get(_id) is not None
    old_binary = database.session.query(BinaryObject).get(_id).binary

    new_storable = create_storable(
        _id=storable.id,
        data=th.Tensor([11, 22, 33, 44]),
        description="NewDummy tensor",
        tags=["new", "dummy", "tensor"],
    )

    storage.__setitem__(storable.id, new_storable)

    new_binary = database.session.query(BinaryObject).get(_id).binary
    assert hash(new_binary) != hash(old_binary)


def test_delitem(client, database, cleanup):
    assert get_metadata(database).length == 0

    storage = DiskObjectStore(database)
    bin_obj = BinaryObject(id=storable.id.value.hex, binary=storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()
    _id = storable.id.value.hex

    assert database.session.query(BinaryObject).get(_id) is not None
    assert get_metadata(database).length == 1

    storage.__delitem__(storable.id)

    assert database.session.query(BinaryObject).get(_id) is None
    assert get_metadata(database).length == 0


def test__len__(client, database, cleanup):
    storage = DiskObjectStore(database)

    bin_obj = BinaryObject(id=storable.id.value.hex, binary=storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    _id = storable.id.value.hex

    assert storage.__len__() == 1
    obj = database.session.query(BinaryObject).get(storable.id.value.hex)
    metadata = get_metadata(database)
    metadata.length -= 1

    database.session.delete(obj)
    database.session.commit()

    assert storage.__len__() == 0


def test_get_keys(client, database, cleanup):
    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    keys = storage.keys()
    assert set(keys) == set([uid1, uid2])


def test_clear(client, database, cleanup):
    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=storable.to_bytes())
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

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    assert storage.__sizeof__() > 0


def test_get_values(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    storable1 = th.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    storable1.describe("Dummy tensor 1")
    storable1.tags = ["dummy1", "tensor"]
    storable1.id = uid1

    storable2 = th.tensor([15.0, 25.0, 35.0, 45.0], requires_grad=True)
    storable2.describe("Dummy tensor 2")
    storable2.tags = ["dummy2", "tensor"]
    storable2.id = uid2

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=storable2.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    binaries = storage.values()

    assert th.eq(binaries[0], storable1).all()
    assert th.eq(binaries[1], storable2).all()


def test_get_objects_of_type(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    storable1 = th.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    storable1.describe("Dummy tensor 1")
    storable1.tags = ["dummy1", "tensor"]
    storable1.id = uid1

    storable2 = th.tensor([15.0, 25.0, 35.0, 45.0], requires_grad=True)
    storable2.describe("Dummy tensor 2")
    storable2.tags = ["dummy2", "tensor"]
    storable2.id = uid2

    new_storable = create_storable(
        _id=storable.id,
        data=th.Tensor([11, 22, 33, 44]),
        description="NewDummy tensor",
        tags=["new", "dummy", "tensor"],
    )

    bin_obj = BinaryObject(id=new_storable.id.value.hex, binary=new_storable.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    bin_obj = BinaryObject(id=uid2.value.hex, binary=storable2.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    binaries = storage.get_objects_of_type(th.Tensor)

    assert th.eq(binaries[0], storable1).all()
    assert th.eq(binaries[1], storable2).all()


def test__getitem__(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    storable1 = th.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    storable1.describe("Dummy tensor 1")
    storable1.tags = ["dummy1", "tensor"]
    storable1.id = uid1

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    retrieved = storage.__getitem__(uid1)
    assert th.eq(retrieved, storable1).all()


def test__getitem__missing(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    storable1 = th.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    storable1.describe("Dummy tensor 1")
    storable1.tags = ["dummy1", "tensor"]
    storable1.id = uid1

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    with pytest.raises(AttributeError):
        retrieved = storage.__getitem__(uid2)


def test_get_object(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    storable1 = th.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    storable1.describe("Dummy tensor 1")
    storable1.tags = ["dummy1", "tensor"]
    storable1.id = uid1

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    retrieved = storage.get_object(uid1)
    assert th.eq(retrieved, storable1).all()


def test_get_object_missing(client, database, cleanup):

    storage = DiskObjectStore(database)
    uid1 = UID()
    uid2 = UID()

    storable1 = th.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    storable1.describe("Dummy tensor 1")
    storable1.tags = ["dummy1", "tensor"]
    storable1.id = uid1

    bin_obj = BinaryObject(id=uid1.value.hex, binary=storable1.to_bytes())
    metadata = get_metadata(database)
    metadata.length += 1
    database.session.add(bin_obj)
    database.session.commit()

    retrieved = storage.get_object(uid2)
    assert retrieved is None
