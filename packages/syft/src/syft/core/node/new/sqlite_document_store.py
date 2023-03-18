# future
from __future__ import annotations

# stdlib
from copy import deepcopy
import os
import sqlite3
import tempfile
import threading
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from typing_extensions import Self

# relative
from .deserialize import _deserialize
from .document_store import BasePartitionSettings
from .document_store import DocumentStore
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .serializable import serializable
from .serialize import _serialize
from .uid import UID


def _repr_debug_(value: Any) -> str:
    if hasattr(value, "_repr_debug_"):
        return value._repr_debug_()
    return repr(value)


def thread_ident() -> int:
    return threading.current_thread().ident


@serializable(recursive_serde=True)
class SQLiteBackingStore(KeyValueBackingStore):
    __attr_state__ = ["index_name", "settings", "store_config"]

    def __init__(
        self,
        index_name: str,
        settings: BasePartitionSettings,
        store_config: StoreConfig,
        ddtype: Optional[type] = None,
    ) -> None:
        self.index_name = index_name
        self.settings = settings
        self.store_config = store_config
        self._ddtype = ddtype
        self._db: Dict[int, sqlite3.Connection] = {}
        self._cur: Dict[int, sqlite3.Connection] = {}
        self.create_table()

    @property
    def table_name(self) -> str:
        return f"{self.settings.name}_{self.index_name}"

    def _connect(self) -> None:
        # SQLite is not thread safe by default so we ensure that each connection
        # comes from a different thread. In cases of Uvicorn and other AWSGI servers
        # there will be many threads handling incoming requests so we need to ensure
        # that different connections are used in each thread. By using a dict for the
        # _db and _cur we can ensure they are never shared
        # print(f"Creating new {type(self)} connection on Thread: {thread_ident()}")
        self.file_path = self.store_config.client_config.file_path
        self._db[thread_ident()] = sqlite3.connect(self.file_path)

    def create_table(self):
        try:
            self.cur.execute(
                f"create table {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, "  # nosec
                + "repr TEXT NOT NULL, value BLOB NOT NULL)"  # nosec
            )
            self.db.commit()
        except sqlite3.OperationalError as e:
            if f"table {self.table_name} already exists" not in str(e):
                raise e

    @property
    def db(self) -> sqlite3.Connection:
        if thread_ident() not in self._db:
            self._connect()
        return self._db[thread_ident()]

    @property
    def cur(self) -> sqlite3.Cursor:
        if thread_ident() not in self._cur:
            self._cur[thread_ident()] = self.db.cursor()

        return self._cur[thread_ident()]

    def _close(self) -> None:
        self._commit()
        self.db.close()

    def _commit(self) -> None:
        self.db.commit()

    def _execute(self, sql: str, *args: Optional[List[Any]]) -> None:
        cursor = self.cur.execute(sql, *args)
        self._commit()
        return cursor

    def _set(self, key: UID, value: Any) -> None:
        if self._exists(key):
            self._update(key, value)
        else:
            insert_sql = f"insert into {self.table_name} (uid, repr, value) VALUES (?, ?, ?)"  # nosec
            data = _serialize(value, to_bytes=True)
            self._execute(insert_sql, [str(key), _repr_debug_(value), data])

    def _update(self, key: UID, value: Any) -> None:
        insert_sql = f"update {self.table_name} set uid = ?, repr = ?, value = ? where uid = ?"  # nosec
        data = _serialize(value, to_bytes=True)
        self._execute(insert_sql, [str(key), _repr_debug_(value), data, str(key)])

    def _get(self, key: UID) -> Any:
        select_sql = f"select * from {self.table_name} where uid = ?"  # nosec
        row = self._execute(select_sql, [str(key)]).fetchone()
        if row is None or len(row) == 0:
            raise KeyError(f"{key} not in {type(self)}")
        data = row[2]
        return _deserialize(data, from_bytes=True)

    def _exists(self, key: UID) -> Any:
        select_sql = f"select uid from {self.table_name} where uid = ?"  # nosec
        row = self._execute(select_sql, [str(key)]).fetchone()
        return bool(row)

    def _get_all(self) -> Any:
        select_sql = f"select * from {self.table_name}"  # nosec
        keys = []
        data = []
        rows = self._execute(select_sql).fetchall()
        for row in rows:
            keys.append(UID(row[0]))
            data.append(_deserialize(row[2], from_bytes=True))
        return dict(zip(keys, data))

    def _get_all_keys(self) -> Any:
        try:
            select_sql = f"select uid from {self.table_name}"  # nosec
            keys = []
            rows = self._execute(select_sql).fetchall()
            for row in rows:
                keys.append(UID(row[0]))
            return keys
        except Exception as e:
            raise e

    def _delete(self, key: UID) -> None:
        select_sql = f"delete from {self.table_name} where uid = ?"  # nosec
        self._execute(select_sql, [str(key)])

    def _delete_all(self) -> None:
        select_sql = f"delete from {self.table_name}"  # nosec
        self._execute(select_sql)

    def _len(self) -> int:
        select_sql = f"select uid from {self.table_name}"  # nosec
        result = self._execute(select_sql)
        if hasattr(result, "__len__"):
            return len(result)
        return 0

    def __setitem__(self, key: Any, value: Any) -> None:
        self._set(key, value)

    def __getitem__(self, key: Any) -> Self:
        try:
            return self._get(key)
        except KeyError as e:
            if self._ddtype is not None:
                return self._ddtype()
            raise e

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
        return self._get_all_keys()

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
    def close(self) -> None:
        self.data._close()
        self.unique_keys._close()
        self.searchable_keys._close()

    def commit(self) -> None:
        self.data._commit()
        self.unique_keys._commit()
        self.searchable_keys._commit()


# the base document store is already a dict but we can change it later
@serializable(recursive_serde=True)
class SQLiteDocumentStore(DocumentStore):
    partition_type = SQLiteStorePartition


@serializable(recursive_serde=True)
class SQLiteStoreClientConfig(StoreClientConfig):
    filename: Optional[str]
    path: Optional[str]

    @property
    def temp_path(self) -> str:
        return tempfile.gettempdir()

    @property
    def file_path(self) -> str:
        path = self.path if self.path else self.temp_path
        return path + os.sep + self.filename


@serializable(recursive_serde=True)
class SQLiteStoreConfig(StoreConfig):
    client_config: StoreClientConfig
    store_type: Type[DocumentStore] = SQLiteDocumentStore
    backing_store: Type[KeyValueBackingStore] = SQLiteBackingStore
