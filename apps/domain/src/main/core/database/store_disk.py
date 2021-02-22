from typing import Optional, Iterable
from json import loads
from os.path import getsize

from torch import Tensor
from loguru import logger
from flask import current_app as app
from syft.core.common.uid import UID
from syft.core.store import ObjectStore, Dataset
from syft.core.common.serde import _deserialize
from syft.core.store.storeable_object import StorableObject

from .bin_storage.bin_obj import BinaryObject
from .bin_storage.metadata import StorageMetadata, get_metadata
from . import db, BaseModel

ENCODING = "UTF-8"


def create_storable(
    _id: UID, data: Tensor, description: str, tags: Iterable[str]
) -> StorableObject:
    obj = StorableObject(id=_id, data=data, description=description, tags=tags)

    return obj


def create_dataset(
    _id: UID, data: Iterable[StorableObject], description: str, tags: Iterable[str]
) -> StorableObject:
    obj = Dataset(id=_id, data=data, description=description, tags=tags)

    return obj


class DiskObjectStore(ObjectStore):
    def __init__(self, db):
        self.db = db

    def __sizeof__(self) -> int:
        uri = app.config["SQLALCHEMY_BINDS"]["bin_store"]
        db_path = uri[10:]
        return getsize(db_path)

    def store(self, obj: Dataset) -> None:
        bin_obj = BinaryObject(id=obj.id.value.hex, binary=obj.to_bytes())
        metadata = get_metadata(self.db)
        metadata.length += 1

        self.db.session.add(bin_obj)
        self.db.session.commit()

    def store_bytes_at(self, key: str, obj: bytes) -> None:
        bin_obj = self.db.session.query(BinaryObject).get(key)
        setattr(bin_obj, "binary", obj)
        self.db.session.commit()

    def store_bytes(self, obj: bytes) -> str:
        _id = UID()
        bin_obj = BinaryObject(id=_id.value.hex, binary=obj)
        metadata = get_metadata(self.db)
        metadata.length += 1

        self.db.session.add(bin_obj)
        self.db.session.commit()
        return _id.value.hex

    def __contains__(self, key: str) -> bool:
        self.db.session.rollback()
        return self.db.session.query(BinaryObject).get(key) is not None

    def __setitem__(self, key: str, value: Dataset) -> None:
        obj = self.db.session.query(BinaryObject).get(key)
        obj.binary = value.to_bytes()
        self.db.session.commit()

    def __getitem__(self, key: str) -> bytes:
        try:
            obj = self.db.session.query(BinaryObject).get(key)
            return obj.binary
        except Exception as e:
            logger.trace(f"{type(self)} get item error {key} {e}")
            raise e

    def get_object(self, key: str) -> Optional[Dataset]:
        obj = None
        if self.db.session.query(BinaryObject).get(key) is not None:
            obj = self.__getitem__(key)
        return obj

    def delete(self, key: str) -> None:
        obj = self.db.session.query(BinaryObject).get(key)
        metadata = get_metadata(self.db)
        metadata.length -= 1

        self.db.session.delete(obj)
        self.db.session.commit()

    def __delitem__(self, key: str) -> None:
        self.delete(key=key)

    def __len__(self) -> int:
        return get_metadata(self.db).length

    def keys(self) -> Iterable[str]:
        keys = self.db.session.query(BinaryObject.id).all()
        keys = [k[0] for k in keys]
        return keys

    def pairs(self):
        ids = self.db.session.query(BinaryObject.id).all()
        return {key[0]: self.get_object(key) for key in ids}

    def clear(self) -> None:
        self.db.session.query(BinaryObject).delete()
        self.db.session.query(StorageMetadata).delete()
        self.db.session.commit()

    def values(self) -> Iterable[Dataset]:
        binaries = self.db.session.query(BinaryObject.binary).all()
        binaries = [_deserialize(blob=b[0], from_bytes=True) for b in binaries]
        return binaries

    def __str__(self) -> str:
        objs = self.db.session.query(BinaryObject).all()
        objs = [obj.__str__() for obj in objs]
        return "{}\n{}".format(get_metadata(self.db).__str__(), objs)
