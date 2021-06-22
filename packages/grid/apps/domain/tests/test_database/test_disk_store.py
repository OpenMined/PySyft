# stdlib
from json import dumps
from json import loads

# third party
import pytest
import numpy as np
import torch as th
from flask import current_app as app
from syft.core.common.uid import UID
from sqlalchemy.exc import NoResultFound
from syft.core.store.storeable_object import StorableObject

from src.main.core.database import *
from src.main.core.database.store_disk import DiskObjectStore

tensor1 = th.tensor([[1, 2, 3, 4], [10, 20, 30, 40]])
tensor2 = th.tensor([[-1, -2, -3, -4], [-100, -200, -300, -400]])


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(BinObject).delete()
        database.session.query(ObjectMetadata).delete()
        database.session.commit()
    except:
        database.session.rollback()


def test__setitem__(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}


def test_delete(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    disk_store.delete(_id)

    assert database.session.query(BinObject).get(str(_id.value)) is None

    with pytest.raises(NoResultFound) as e_info:
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()


def test__getitem__success(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    retrieved = disk_store.__getitem__(_id)
    assert th.all(th.eq(retrieved.data, tensor1))
    assert retrieved.id == _id


def test__getitem__fail(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    new_id = UID()
    with pytest.raises(Exception) as e_info:
        retrieved = disk_store.__getitem__(new_id)


def test__contains__success(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    retrieved = disk_store.__contains__(_id)
    assert retrieved


def test__contains__fail(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()

    retrieved = disk_store.__contains__(_id)
    assert not retrieved


def test__len__(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)

    assert disk_store.__len__() == 0

    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    assert disk_store.__len__() == 1


def test_get_object_success(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    retrieved = disk_store.get_object(_id)
    assert th.all(th.eq(retrieved.data, tensor1))
    assert retrieved.id == _id


def test_get_object_fail(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    new_id = UID()
    retrieved = disk_store.get_object(new_id)
    assert retrieved is None


def test_clear(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    _id = UID()
    storable = StorableObject(id=_id, data=tensor1)
    disk_store.__setitem__(_id, storable)

    bin_obj = database.session.query(BinObject).get(str(_id.value))
    metadata = (
        database.session.query(ObjectMetadata).filter_by(obj=str(_id.value)).one()
    )

    assert bin_obj is not None
    assert th.all(th.eq(bin_obj.object, tensor1))

    assert metadata is not None
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.read_permissions == {}
    assert metadata.search_permissions == {}

    retrieved = disk_store.get_object(_id)
    assert th.all(th.eq(retrieved.data, tensor1))
    assert retrieved.id == _id

    disk_store.clear()

    assert database.session.query(BinObject).count() == 0
    assert database.session.query(ObjectMetadata).count() == 0


def test_get_objects_of_type(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    a_id = UID()

    id1 = UID()
    id2 = UID()
    storable1 = StorableObject(id=id1, data=tensor1)
    disk_store.__setitem__(id1, storable1)
    storable2 = StorableObject(id=id2, data=tensor2)
    disk_store.__setitem__(id2, storable2)

    selected = disk_store.get_objects_of_type(th.Tensor)
    selected_data = [x.data for x in selected]

    assert any(th.all(th.eq(tensor1, d_)) for d_ in selected_data)
    assert any(th.all(th.eq(tensor2, d_)) for d_ in selected_data)
    assert len(selected_data) == 2


def test_keys(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    a_id = UID()

    id1 = UID()
    id2 = UID()
    storable1 = StorableObject(id=id1, data=tensor1)
    disk_store.__setitem__(id1, storable1)
    storable2 = StorableObject(id=id2, data=tensor2)
    disk_store.__setitem__(id2, storable2)

    keys = disk_store.keys()
    assert any(id1 == k for k in keys)
    assert any(id2 == k for k in keys)
    assert len(keys) == 2


def test_values(client, database, cleanup):
    disk_store = DiskObjectStore(database)
    a_id = UID()

    id1 = UID()
    id2 = UID()
    storable1 = StorableObject(id=id1, data=tensor1)
    disk_store.__setitem__(id1, storable1)
    storable2 = StorableObject(id=id2, data=tensor2)
    disk_store.__setitem__(id2, storable2)

    values = disk_store.values()
    values_data = [v.data for v in values]

    assert any(th.all(th.eq(tensor1, v)) for v in values_data)
    assert any(th.all(th.eq(tensor2, v)) for v in values_data)
    assert len(values_data) == 2
