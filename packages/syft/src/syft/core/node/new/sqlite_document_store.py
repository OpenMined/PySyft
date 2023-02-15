# future
from __future__ import annotations

# stdlib
from copy import deepcopy
import sqlite3
from typing import Any
from typing import List
from typing import Optional
from typing import Type

# third party
from typing_extensions import Self

# relative
from ...common.serde.deserialize import _deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize
from ...common.uid import UID
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition


class SQLiteBackingStore(KeyValueBackingStore):
    def __init__(self, settings: PartitionSettings) -> None:
        self.settings = settings
        self.table_name = self.settings.name
        self.db = sqlite3.connect("/tmp/db.sqlite")
        self.cur = self.db.cursor()
        try:
            self.cur.execute(
                f"create table {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, value BLOB NOT NULL)"
            )
            self.db.commit()
        except sqlite3.OperationalError as e:
            if f"table {self.table_name} already exists" not in str(e):
                raise e

    def _execute(self, sql: str, *args: Optional[List[Any]]) -> None:
        cursor = self.cur.execute(sql, *args)
        self.db.commit()
        return cursor

    def _set(self, key: UID, value: Any) -> None:
        try:
            if self._exists(key):
                self.update(key, value)
            else:
                insert_sql = f"insert into {self.table_name} (uid, value) VALUES (?, ?)"
                data = _serialize(value, to_bytes=True)
                self._execute(insert_sql, [str(key), data])
        except Exception as e:
            print("Failed to _set", e)
            raise e

    def _update(self, key: UID, value: Any) -> None:
        insert_sql = f"update {self.table_name} set uid = ?, value = ? where uid = ?"
        data = _serialize(value, to_bytes=True)
        self._execute(insert_sql, [str(key), data, str(key)])

    def _get(self, key: UID) -> Any:
        select_sql = f"select * from {self.table_name} where uid = ?"
        row = self._execute(select_sql, [str(key)]).fetchone()
        data = row[1]
        return _deserialize(data, from_bytes=True)

    def _exists(self, key: UID) -> Any:
        select_sql = f"select uid from {self.table_name} where uid = ?"
        row = self._execute(select_sql, [str(key)]).fetchone()
        return bool(row)

    def _get_all(self) -> Any:
        try:
            select_sql = f"select * from {self.table_name}"
            keys = []
            data = []
            rows = self._execute(select_sql).fetchall()
            for row in rows:
                keys.append(UID(row[0]))
                data.append(_deserialize(row[1], from_bytes=True))
            return dict(zip(keys, data))
        except Exception as e:
            print("Failed to _get_all", e)
            raise e

    def _delete(self, key: UID) -> None:
        select_sql = f"delete from {self.table_name} where uid = ?"
        self._execute(select_sql, [str(key)])

    def _delete_all(self) -> None:
        select_sql = f"delete from {self.table_name}"
        self._execute(select_sql)

    def _len(self) -> int:
        select_sql = f"select uid from {self.table_name}"
        return len(self._execute(select_sql))

    def __setitem__(self, key: Any, value: Any) -> None:
        self._set(key, value)

    def __getitem__(self, key: Any) -> Self:
        return self._get(key)

    def __repr__(self) -> str:
        return repr(self._get_all())

    def __len__(self) -> int:
        return self._len()

    def __delitem__(self, key: str):
        self._delete(key)

    def clear(self) -> Self:
        self._delete_all()

    def copy(self) -> Self:
        return deepcopy(self)

    def keys(self) -> Any:
        return self._get_all().keys()

    def values(self) -> Any:
        return self._get_all().values()

    def items(self) -> Any:
        return self._get_all().items()

    def pop(self, key: Any) -> Self:
        value = self._get(key)
        self._delete(key)
        return value

    def __contains__(self, key: Any) -> bool:
        return self._exists(key)

    def __iter__(self) -> Any:
        return iter(self.keys())


@serializable(recursive_serde=True)
class SQLiteStorePartition(KeyValueStorePartition):
    pass


# the base document store is already a dict but we can change it later
@serializable(recursive_serde=True)
class SQLiteDocumentStore(DocumentStore):
    partition_type = SQLiteStorePartition


@serializable(recursive_serde=True)
class SQLiteStoreConfig(StoreConfig):
    store_type: Type[DocumentStore] = SQLiteDocumentStore
    backing_store: Type[KeyValueBackingStore] = SQLiteBackingStore
