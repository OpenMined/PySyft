from typing import Optional, Iterable
from os.path import getsize

from torch import Tensor
from loguru import logger
from flask import current_app as app
from syft.core.common.uid import UID
from syft.core.store import ObjectStore
from syft.core.common.serde import _deserialize
from syft.core.store.storeable_object import StorableObject

from .bin_storage.bin_obj import BinaryObject
from .bin_storage.metadata import StorageMetadata, get_metadata
from . import db, BaseModel

ENCODING = "UTF-8"


# from main.core.database.bin_storage.metadata import *
def create_storable(
    _id: UID, data: Tensor, description: str, tags: Iterable[str]
) -> StorableObject:
    obj = StorableObject(id=_id, data=data, description=description, tags=tags)

    return obj


class DiskObjectStore(ObjectStore):
    def __init__(self, db):
        self.db = db

    def __sizeof__(self) -> int:
        uri = app.config["SQLALCHEMY_BINDS"]["bin_store"]
        db_path = uri[10:]
        return getsize(db_path)

    def store(self, obj: StorableObject) -> None:
        bin_obj = BinaryObject(id=obj.id.value.hex, binary=obj.to_bytes())
        metadata = get_metadata(self.db)
        metadata.length += 1

        self.db.session.add(bin_obj)
        self.db.session.commit()

    def __contains__(self, item: UID) -> bool:
        self.db.session.rollback()
        _id = item.value.hex
        return self.db.session.query(BinaryObject).get(_id) is not None

    def __setitem__(self, key: UID, value: StorableObject) -> None:
        obj = self.db.session.query(BinaryObject).get(key.value.hex)
        obj.binary = value.to_bytes()
        self.db.session.commit()

    def __getitem__(self, key: UID) -> StorableObject:
        try:
            obj = self.db.session.query(BinaryObject).get(key.value.hex)
            obj = _deserialize(blob=obj.binary, from_bytes=True)
            return obj
        except Exception as e:
            logger.trace(f"{type(self)} get item error {key} {e}")
            raise e

    def get_object(self, key: UID) -> Optional[StorableObject]:
        obj = None
        if self.db.session.query(BinaryObject).get(key.value.hex) is not None:
            obj = self.__getitem__(key)
        return obj

    def delete(self, key: UID) -> None:
        obj = self.db.session.query(BinaryObject).get(key.value.hex)
        metadata = get_metadata(self.db)
        metadata.length -= 1

        self.db.session.delete(obj)
        self.db.session.commit()

    def __delitem__(self, key: UID) -> None:
        self.delete(key=key)

    def __len__(self) -> int:
        return get_metadata(self.db).length

    def keys(self) -> Iterable[UID]:
        ids = self.db.session.query(BinaryObject.id).all()
        return [UID.from_string(value=_id[0]) for _id in ids]

    def clear(self) -> None:
        self.db.session.query(BinaryObject).delete()
        self.db.session.query(StorageMetadata).delete()
        self.db.session.commit()

    def values(self) -> Iterable[StorableObject]:
        # TODO _deserialize creates storable with no data or tags for StorableObject
        binaries = self.db.session.query(BinaryObject.binary).all()
        binaries = [_deserialize(blob=b[0], from_bytes=True) for b in binaries]
        return binaries

    def get_objects_of_type(self, obj_type: type) -> Iterable[StorableObject]:
        return [v for v in self.values() if isinstance(v, obj_type)]

    def __str__(self) -> str:
        objs = self.db.session.query(BinaryObject).all()
        objs = [obj.__str__() for obj in objs]
        return "{}\n{}".format(get_metadata(self.db).__str__(), objs)
