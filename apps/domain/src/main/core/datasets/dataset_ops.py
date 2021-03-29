from typing import Optional, Iterable
from copy import deepcopy
from io import StringIO

import torch as th
import numpy as np
import pandas as pd
from syft.core.common.uid import UID
from syft.core.store.storeable_object import StorableObject

from ..database import db
from ..database.bin_storage.bin_obj import BinObject, ObjectMetadata
from ..database.bin_storage.json_obj import JsonObject
from ..database.store_disk import DiskObjectStore
from ..database.bin_storage.metadata import get_metadata
from ..database.dataset.datasetgroup import DatasetGroup


def create_dataset(df_json: dict) -> dict:
    _json = deepcopy(df_json)
    storage = DiskObjectStore(db)
    mapping = []

    # Separate CSV from metadata
    for el in _json["tensors"].copy():
        _id = UID()
        _json["tensors"][el]["id"] = str(_id.value)
        mapping.append((el, _id, _json["tensors"][el].pop("content", None)))

    # Ensure we have same ID in metadata and dataset
    df_id = UID()
    _json["id"] = str(df_id.value)

    # Create storables from UID/CSV. Update metadata
    storables = []
    for idx, (name, _id, raw_file) in enumerate(mapping):
        _tensor = pd.read_csv(StringIO(raw_file))
        _tensor = th.tensor(_tensor.values.astype(np.float32))

        _json["tensors"][name]["shape"] = [int(x) for x in _tensor.size()]
        _json["tensors"][name]["dtype"] = "{}".format(_tensor.dtype)
        storage.__setitem__(_id, StorableObject(id=_id, data=_tensor))
        # Ensure we have same ID in metadata and dataset
        db.session.add(
            DatasetGroup(bin_object=str(_id.value), dataset=str(df_id.value))
        )

    json_obj = JsonObject(id=_json["id"], binary=_json)
    metadata = get_metadata(db)
    metadata.length += 1

    db.session.add(json_obj)
    db.session.commit()
    return _json


def get_dataset_metadata(key: str) -> Optional[dict]:
    obj = db.session.query(JsonObject).get(key)
    if obj is not None:
        obj = obj.binary
    return obj


def get_all_datasets_metadata():
    ids = db.session.query(JsonObject.id).all()
    return [get_dataset_metadata(key) for key in ids]


def update_dataset(key: str, df_json: dict) -> dict:
    _json = deepcopy(df_json)
    storage = DiskObjectStore(db)

    json_obj = db.session.query(JsonObject).get(key)
    past_json = json_obj.binary
    past_ids = [x["id"] for x in past_json["tensors"].values()]

    mapping = []
    # Separate CSV from metadata
    for el in _json["tensors"].copy():
        if (
            _json["tensors"][el].get("id", None) is not None
            and _json["tensors"][el].get("id", None) in past_ids
        ):
            _json["tensors"][el]["id"] = past_json["tensors"][el]["id"]
        else:
            _id = UID()
            _json["tensors"][el]["id"] = str(_id.value)
        mapping.append((el, _id, _json["tensors"][el].pop("content", None)))

    # Ensure we have same ID in metadata and dataset
    df_id = past_json["id"]
    _json["id"] = df_id

    # Clean existing storables in storage
    db.session.query(DatasetGroup).filter_by(dataset=df_id).delete(
        synchronize_session=False
    )
    for key in past_ids:
        storage.delete(key)

    # Create storables from UID/CSV. Update metadata
    storables = []
    for idx, (name, _id, raw_file) in enumerate(mapping):
        _tensor = pd.read_csv(StringIO(raw_file))
        _tensor = th.tensor(_tensor.values.astype(np.float32))

        _json["tensors"][name]["shape"] = [int(x) for x in _tensor.size()]
        _json["tensors"][name]["dtype"] = "{}".format(_tensor.dtype)
        storage.__setitem__(_id, StorableObject(id=_id, data=_tensor))
        # Ensure we have same ID in metadata and dataset
        db.session.add(DatasetGroup(bin_object=str(_id.value), dataset=df_id))

    setattr(json_obj, "binary", _json)
    db.session.commit()
    return _json


def delete_dataset(key: str) -> None:
    storage = DiskObjectStore(db)
    ids = db.session.query(DatasetGroup.bin_object).filter_by(dataset=key).all()
    ids = [x[0] for x in ids]

    for _id in ids:
        bin_obj = db.session.query(BinObject).filter_by(id=_id).first()
        metadata = db.session.query(ObjectMetadata).filter_by(obj=key).first()

        if bin_obj is not None:
            db.session.delete(bin_obj)
        if metadata is not None:
            db.session.delete(metadata)

    db.session.query(DatasetGroup).filter_by(dataset=key).delete(
        synchronize_session=False
    )
    json_obj = db.session.query(JsonObject).get(key)
    metadata = get_metadata(db)
    metadata.length -= 1

    db.session.delete(json_obj)
    db.session.commit()
