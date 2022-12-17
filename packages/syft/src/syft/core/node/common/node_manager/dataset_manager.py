# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import cast
from uuid import UUID

# relative
from ..... import serialize
from ....common.uid import UID
from ..node_table.dataset import NoSQLBinObjDataset
from ..node_table.dataset import NoSQLDataset
from .database_manager import NoSQLDatabaseManager


class DatasetNotFoundError(Exception):
    pass


class NoSQLDatasetManager(NoSQLDatabaseManager):
    """Class to manage dataset database actions."""

    _collection_name = "datasets"
    __canonical_object_name__ = "Dataset"

    def first(self, **kwargs: Any) -> NoSQLDataset:
        result = super().find_one(kwargs)
        if not result:
            raise DatasetNotFoundError
        return result

    def register(self, **kwargs: Any) -> UID:
        """Register e  new object into the database.

        Args:
            parameters : List of object parameters.
        Returns:
            object: Database Object
        """
        tags = list(map(str, kwargs.get("tags", [])))
        manifest = str(kwargs.get("manifest", ""))
        name = str(kwargs.get("name", ""))
        description = str(kwargs.get("description", ""))

        print("All the dataset arguments:")
        print(kwargs)
        blob_metadata = {}
        str_metadata = {}
        for key, value in kwargs.items():
            if (
                key != "tags"
                and key != "manifest"
                and key != "description"
                and key != "name"
            ):
                if isinstance(key, str) and isinstance(value, str):
                    str_metadata[key] = value
                else:
                    blob_metadata[str(key)] = cast(
                        bytes, serialize(value, to_bytes=True)
                    ).hex()

        _obj = NoSQLDataset(
            id=UID(),
            name=name,
            tags=tags,
            manifest=manifest,
            description=description,
            str_metadata=str_metadata,
            blob_metadata=blob_metadata,
        )
        self.add(_obj)
        return _obj.id

    def add_obj(
        self,
        name: str,
        dataset_id: Union[UID, UUID, str],
        obj_id: Union[UID, UUID, str],
        dtype: str,
        shape: str,
    ) -> None:
        dataset_uid: UID = UID._check_or_convert(dataset_id)
        dataset = self.first(id=dataset_uid)
        obj_dataset_relation = NoSQLBinObjDataset(
            name=name,
            dataset=dataset_uid,
            obj_id=obj_id,
            dtype=dtype,
            shape=shape,
        )
        dataset.bin_obj_dataset.append(obj_dataset_relation)

        self.update_one({"id": dataset_uid}, {"__blob__": dataset.to_bytes()})

    def get(
        self, dataset_id: Union[UUID, UID, str]
    ) -> Tuple[NoSQLDataset, List[NoSQLBinObjDataset]]:
        dataset_uid = UID._check_or_convert(dataset_id)
        dataset = self.first(id=dataset_uid)

        return dataset, dataset.bin_obj_dataset

    def delete_bin_obj(self, bin_obj_id: Union[UID, UUID, str]) -> None:
        obj_id = UID._check_or_convert(bin_obj_id)
        _flag = False  # checking for first occurence
        bin_obj_index = -1
        for dataset in self.all():
            for idx, bin_obj in enumerate(dataset.bin_obj_dataset):
                if bin_obj.obj_id == obj_id:
                    bin_obj_index = idx
                    _flag = True
                    break
            if _flag:
                del dataset.bin_obj_dataset[bin_obj_index]
                self.update_one({"id": dataset.id}, {"__blob__": dataset.to_bytes()})
                break

    def set(self, dataset_id: str, metadata: Dict) -> None:
        self.update({"id": dataset_id}, metadata)

    def all(self) -> List[NoSQLDataset]:
        return super().all()
