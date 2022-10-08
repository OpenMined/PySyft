# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import cast
from uuid import uuid4

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ..... import serialize
from ....common.uid import UID
from ..node_table.bin_obj_dataset import BinObjDataset
from ..node_table.oblv_keys import OblvKeys
from .database_manager import DatabaseManager


class OblvKeyManager(DatabaseManager):
    schema = OblvKeys

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=OblvKeyManager.schema, db=database)

    def add(
        self, public_key: bytes, private_key: bytes
    ) -> None:

        key_obj = OblvKeys(
            id=str(UID().value),
            public_key=public_key,
            private_key=private_key,
        )
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(key_obj)
        session_local.commit()
        session_local.close()

    def get(self) -> OblvKeys:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        res = session_local.query(OblvKeys).first()
        session_local.close()
        return res
        
    def remove(self) -> None:
        try:
            session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
            data_to_delete = (
                session_local.query(OblvKeys).first()
            )
            session_local.delete(data_to_delete)
            session_local.commit()
            session_local.close()
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error . {e}")
