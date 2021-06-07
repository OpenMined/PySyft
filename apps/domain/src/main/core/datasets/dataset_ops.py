# stdlib
from copy import deepcopy
import csv
from io import StringIO
import tarfile
from typing import Iterable
from typing import Optional

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
import numpy as np
import pandas as pd
from pandas import DataFrame
from syft.core.common.group import VERIFYALL
from syft.core.common.group import VerifyAll
from syft.core.common.uid import UID
from syft.core.node.common.action.save_object_action import SaveObjectAction
from syft.core.store.storeable_object import StorableObject
import torch as th

# grid relative
from ..database import db
from ..database.bin_storage.bin_obj import BinObject
from ..database.bin_storage.bin_obj import ObjectMetadata
from ..database.bin_storage.json_obj import JsonObject
from ..database.bin_storage.metadata import get_metadata
from ..database.dataset.datasetgroup import BinObjDataset
from ..database.dataset.datasetgroup import Dataset
from ..database.dataset.datasetgroup import DatasetGroup
from ..database.store_disk import DiskObjectStore
from ..database.utils import model_to_json


def decompress(file_obj):
    tar_obj = tarfile.open(fileobj=file_obj)
    tar_obj.extractall()
    return tar_obj


def extract_metadata_info(tar_obj):
    tags = []
    manifest = ""
    description = ""
    skip_files = []
    for file_obj in tar_obj.members:
        if "tags" in file_obj.name:
            tags = tar_obj.extractfile(file_obj.name).read().decode().split("\n")[:-1]
            skip_files.append(file_obj.name)
        elif "description" in file_obj.name:
            description = tar_obj.extractfile(file_obj.name).read().decode()
            skip_files.append(file_obj.name)
        elif "manifest" in file_obj.name:
            manifest = tar_obj.extractfile(file_obj.name).read().decode()
            skip_files.append(file_obj.name)

    return tags, manifest, description, skip_files


def process_items(node, tar_obj, user_key):
    # Optional fields
    tags, manifest, description, skip_files = extract_metadata_info(tar_obj)

    dataset_db = Dataset(
        id=str(UID().value), manifest=manifest, description=description, tags=tags
    )
    db.session.add(dataset_db)
    data = list()
    for item in tar_obj.members:
        if not item.isdir() and (not item.name in skip_files):
            reader = csv.reader(
                tar_obj.extractfile(item.name).read().decode().split("\n"),
                delimiter=",",
            )

            dataset = []

            for row in reader:
                if len(row) != 0:
                    dataset.append(row)
            dataset = np.array(dataset, dtype=np.float)
            df = th.tensor(dataset, dtype=th.float32)
            id_at_location = UID()

            # Step 2: create message which contains object to send
            storable = StorableObject(
                id=id_at_location,
                data=df,
                tags=tags + ["#" + item.name.split("/")[-1]],
                search_permissions={VERIFYALL: None},
            )

            obj_msg = SaveObjectAction(obj=storable, address=node.address)

            signed_message = obj_msg.sign(
                signing_key=SigningKey(user_key.encode("utf-8"), encoder=HexEncoder)
            )

            node.recv_immediate_msg_without_reply(msg=signed_message)

            obj_dataset_relation = BinObjDataset(
                name=item.name,
                dataset=dataset_db.id,
                obj=str(id_at_location.value),
                dtype=df.__class__.__name__,
                shape=str(tuple(df.shape)),
            )
            db.session.add(obj_dataset_relation)
            data.append(
                {
                    "name": obj_dataset_relation.name,
                    "id": str(id_at_location.value),
                    "tags": tags + ["#" + item.name.split("/")[-1]],
                    "dtype": obj_dataset_relation.dtype,
                    "shape": obj_dataset_relation.shape,
                }
            )

    db.session.commit()
    ds = model_to_json(dataset_db)
    ds["data"] = data
    return ds


def create_df_dataset(node, tarfile, key):
    try:
        tar_obj = decompress(tarfile)
        response = process_items(node, tar_obj, key)
        return response, 200
    except Exception as e:
        return {"error": str(e)}, 400


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


def get_all_datasets():
    return list(db.session.query(Dataset).all())


def get_all_relations(key):
    return list(db.session.query(BinObjDataset).filter_by(dataset=key).all())


def get_specific_dataset_and_relations(key):
    ds = db.session.query(Dataset).filter_by(id=key).first()
    objs = get_all_relations(key)
    return ds, objs


def get_all_datasets_metadata():
    ids = db.session.query(JsonObject.id).all()
    return [get_dataset_metadata(key) for key in ids]


def update_dataset_metadata(key: str, **kwargs) -> None:
    json_obj = db.session.query(JsonObject).get(key)

    if json_obj is None:
        return

    _json = deepcopy(json_obj.binary)

    for att, value in kwargs.items():
        if att not in _json:
            _json[att] = {value["verify_key"]: value["request_id"]}
        else:
            _json[att][value["verify_key"]] = value["request_id"]

    setattr(json_obj, "binary", _json)
    db.session.commit()


def update_dataset(key: str, tags: list, manifest: str, description: str):
    if tags:
        db.session.query(Dataset).filter_by(id=key).update({"tags": tags})
    elif manifest:
        db.session.query(Dataset).filter_by(id=key).update({"manifest": manifest})
    elif description:
        db.session.query(Dataset).filter_by(id=key).update({"description": description})

    db.session.commit()


def delete_dataset(key: str) -> None:
    ds_objs = get_all_relations(key)
    for ds_obj in ds_objs:
        db.session.query(BinObject).filter_by(id=ds_obj.obj).delete()
        db.session.query(ObjectMetadata).filter_by(obj=ds_obj.obj).delete()
        db.session.delete(ds_obj)

    db.session.query(Dataset).filter_by(id=key).delete()
    db.session.commit()
