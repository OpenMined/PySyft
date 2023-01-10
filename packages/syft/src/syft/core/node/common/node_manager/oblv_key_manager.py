# relative
from ..exceptions import OblvKeyNotFoundError
from ..node_table.oblv_keys import NoSQLOblvKeys
from .database_manager import NoSQLDatabaseManager


class NoSQLOblvKeyManager(NoSQLDatabaseManager):
    _collection_name = "oblv_keys"
    __canonical_object_name__ = "OblvKeys"

    def add_keys(self, public_key: bytes, private_key: bytes) -> None:

        key_obj = NoSQLOblvKeys(
            public_key=public_key,
            private_key=private_key,
        )

        self.add(key_obj)

    def get(self) -> NoSQLOblvKeys:
        result = super().all()
        if not result:
            raise OblvKeyNotFoundError
        return result[0]

    def remove(self) -> None:
        super().clear()
