# stdlib
from copy import deepcopy
from io import StringIO
from json import loads
from os.path import getsize
from typing import Iterable
from typing import Optional

# third party
from flask import current_app as app
from loguru import logger
import numpy as np
import pandas as pd
from syft import deserialize
from syft import serialize
from syft.core.common.uid import UID
from syft.core.store import Dataset
from syft.core.store import ObjectStore
from syft.core.store.storeable_object import StorableObject
import torch as th
from torch import Tensor

# grid relative
from . import BaseModel
from . import db
from .bin_storage.bin_obj import BinaryObject
from .bin_storage.json_obj import JsonObject
from .bin_storage.metadata import StorageMetadata
from .bin_storage.metadata import get_metadata

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


def dataset_to_dict(dataset: Dataset) -> dict:
    _dict = {}
    _dict["id"] = dataset.id.value.hex
    _dict["tags"] = dataset.tags
    _dict["description"] = dataset.description
    _dict["read_permissions"] = dataset.read_permissions
    _dict["search_permissions"] = dataset.search_permissions
    return _dict


class DiskObjectStore(ObjectStore):
    def __init__(self, db):
        self.db = db

    def __sizeof__(self) -> int:
        uri = app.config["SQLALCHEMY_BINDS"]["bin_store"]
        db_path = uri[10:]
        return getsize(db_path)

    def store(self, obj: Dataset, _json: dict) -> None:
        bin_obj = BinaryObject(
            id=obj.id.value.hex, binary=serialize(obj, to_bytes=True)
        )
        json_obj = JsonObject(id=_json["id"], binary=_json)
        metadata = get_metadata(self.db)
        metadata.length += 1

        self.db.session.add(bin_obj)
        self.db.session.add(json_obj)
        self.db.session.commit()

    def store_json(self, df_json: dict) -> dict:
        _json = deepcopy(df_json)
        mapping = []
        # Separate CSV from metadata
        for el in _json["tensors"].copy():
            _id = UID()
            _json["tensors"][el]["id"] = _id.value.hex
            mapping.append((el, _id, _json["tensors"][el].pop("content", None)))

        # Create storables from UID/CSV
        # Update metadata
        storables = []
        for idx, (name, _id, raw_file) in enumerate(mapping):
            _tensor = pd.read_csv(StringIO(raw_file))
            _tensor = th.tensor(_tensor.values.astype(np.float32))

            _json["tensors"][name]["shape"] = [int(x) for x in _tensor.size()]
            _json["tensors"][name]["dtype"] = "{}".format(_tensor.dtype)
            storables.append(StorableObject(id=_id, data=_tensor))

        # Ensure we have same ID in metadata and dataset
        _id = UID()
        df = Dataset(id=_id, data=storables)
        _json["id"] = _id.value.hex

        bin_obj = BinaryObject(id=df.id.value.hex, binary=serialize(df, to_bytes=True))
        json_obj = JsonObject(id=_json["id"], binary=_json)
        metadata = get_metadata(self.db)
        metadata.length += 1

        self.db.session.add(bin_obj)
        self.db.session.add(json_obj)
        self.db.session.commit()
        return _json

    def update_dataset(self, key: str, df_json: dict) -> dict:
        _json = deepcopy(df_json)
        json_obj = self.db.session.query(JsonObject).get(key)
        bin_obj = self.db.session.query(BinaryObject).get(key)

        mapping = []
        # Separate CSV from metadata
        for el in _json["tensors"].copy():
            _id = UID()
            _json["tensors"][el]["id"] = _id.value.hex
            mapping.append((el, _id, _json["tensors"][el].pop("content", None)))

        # Create storables from UID/CSV
        # Update metadata
        storables = []
        for idx, (name, _id, raw_file) in enumerate(mapping):
            _tensor = pd.read_csv(StringIO(raw_file))
            _tensor = th.tensor(_tensor.values.astype(np.float32))

            _json["tensors"][name]["shape"] = [int(x) for x in _tensor.size()]
            _json["tensors"][name]["dtype"] = "{}".format(_tensor.dtype)
            storables.append(StorableObject(id=_id, data=_tensor))

        # Ensure we have same ID in metadata and dataset
        _id = json_obj.id
        _id = UID.from_string(_id)
        df = Dataset(id=_id, data=storables)
        _json["id"] = _id.value.hex

        metadata = get_metadata(self.db)
        metadata.length += 1

        setattr(bin_obj, "binary", serialize(df, to_bytes=True))
        setattr(json_obj, "binary", _json)
        self.db.session.commit()
        return _json

    def update_dataset_metadata(self, key: str, **kwargs) -> None:
        json_obj = self.db.session.query(JsonObject).get(key)

        if json_obj is None:
            return

        _json = deepcopy(json_obj.binary)

        for att, value in kwargs.items():
            if att not in _json:
                _json[att] = {value["verify_key"]: value["request_id"]}
            else:
                _json[att][value["verify_key"]] = value["request_id"]

        setattr(json_obj, "binary", _json)
        self.db.session.commit()

    def store_bytes_at(self, key: str, obj: bytes) -> None:
        bin_obj = self.db.session.query(BinaryObject).get(key)

        dataset = deserialize(blob=obj, from_bytes=True)
        json_obj = self.db.session.query(JsonObject).get(key)
        _json = dataset_to_dict(dataset)

        setattr(bin_obj, "binary", obj)
        setattr(json_obj, "binary", _json)
        self.db.session.commit()

    def store_bytes(self, obj: bytes) -> str:
        _id = UID()
        bin_obj = BinaryObject(id=_id.value.hex, binary=obj)

        dataset = deserialize(blob=obj, from_bytes=True)
        json_obj = dataset_to_dict(dataset)
        json_obj = JsonObject(id=_id.value.hex, binary=json_obj)

        metadata = get_metadata(self.db)
        metadata.length += 1

        self.db.session.add(bin_obj)
        self.db.session.add(json_obj)
        self.db.session.commit()
        return _id.value.hex

    def __contains__(self, key: str) -> bool:
        self.db.session.rollback()
        return self.db.session.query(BinaryObject).get(key) is not None

    def __setitem__(self, key: str, value: Dataset) -> None:
        obj = self.db.session.query(BinaryObject).get(key)
        obj.binary = serialize(value, to_bytes=True)
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

    def get_dataset_metadata(self, key: str) -> Optional[dict]:
        obj = self.db.session.query(JsonObject).get(key)
        if obj is not None:
            obj = obj.binary
        return obj

    def delete(self, key: str) -> None:
        obj = self.db.session.query(BinaryObject).get(key)
        json_obj = self.db.session.query(JsonObject).get(key)
        metadata = get_metadata(self.db)
        metadata.length -= 1

        self.db.session.delete(obj)
        self.db.session.delete(json_obj)
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

    def get_all_datasets_metadata(self):
        ids = self.db.session.query(JsonObject.id).all()
        return [self.get_dataset_metadata(key) for key in ids]

    def clear(self) -> None:
        self.db.session.query(BinaryObject).delete()
        self.db.session.query(StorageMetadata).delete()
        self.db.session.commit()

    def values(self) -> Iterable[Dataset]:
        binaries = self.db.session.query(BinaryObject.binary).all()
        binaries = [deserialize(blob=b[0], from_bytes=True) for b in binaries]
        return binaries

    def __str__(self) -> str:
        objs = self.db.session.query(BinaryObject).all()
        objs = [obj.__str__() for obj in objs]
        return "{}\n{}".format(get_metadata(self.db).__str__(), objs)
