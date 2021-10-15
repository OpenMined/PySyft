# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import cast

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ..... import serialize
from ....common.uid import UID
from ..node_table.bin_obj_dataset import BinObjDataset
from ..node_table.dataset import Dataset
from .database_manager import DatabaseManager


class DatasetManager(DatabaseManager):

    schema = Dataset

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=DatasetManager.schema, db=database)

    def register(self, **kwargs: Any) -> str:
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

        _obj = self._schema(
            id=str(UID().value),
            name=name,
            tags=tags,
            manifest=manifest,
            description=description,
            str_metadata=str_metadata,
            blob_metadata=blob_metadata,
        )
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        obj_id = _obj.id
        session_local.commit()
        session_local.close()
        return obj_id

    def add(
        self, name: str, dataset_id: str, obj_id: str, dtype: str, shape: str
    ) -> None:

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

    def get(self, dataset_id: str) -> Tuple[Dataset, List[bytes]]:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        ds = session_local.query(Dataset).filter_by(id=dataset_id).first()
        objs = list(
            session_local.query(BinObjDataset).filter_by(dataset=dataset_id).all()
        )
        session_local.close()
        return ds, objs

    def set(self, dataset_id: str, metadata: Dict) -> None:
        self.modify({"id": dataset_id}, metadata)
