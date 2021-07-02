# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
from sqlalchemy.orm import sessionmaker

# relative
from ....common.uid import UID
from ..tables.bin_obj_dataset import BinObjDataset
from ..tables.dataset import Dataset
from .database_manager import DatabaseManager


class DatasetManager(DatabaseManager):

    schema = Dataset

    def __init__(self, database):
        self._schema = DatasetManager.schema
        self.db = database

    def register(self, **kwargs) -> Any:
        """Register e  new object into the database.

        Args:
            parameters : List of object parameters.
        Returns:
            object: Database Object
        """
        tags = list(map(str, kwargs.get("tags", [])))
        manifest = str(kwargs.get("manifest", ""))
        description = str(kwargs.get("description", ""))

        _obj = self._schema(
            id=str(UID().value), tags=tags, manifest=manifest, description=description
        )
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        obj_id = _obj.id
        session_local.commit()
        session_local.close()
        return obj_id

    def add(self, name: str, dataset_id: int, obj_id: str, dtype: str, shape: str):

        obj_dataset_relation = BinObjDataset(
            name=name,
            dataset=dataset_id,
            obj=obj_id,
            dtype=dtype,
            shape=shape,
        )
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(obj_dataset_relation)
        session_local.commit()
        session_local.close()

    def get(self, dataset_id: str) -> dict:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        ds = session_local.query(Dataset).filter_by(id=dataset_id).first()
        objs = list(
            session_local.query(BinObjDataset).filter_by(dataset=dataset_id).all()
        )
        session_local.close()
        return ds, objs

    def set(self, dataset_id: str, metadata: Dict):
        self.modify({"id": dataset_id}, metadata)
