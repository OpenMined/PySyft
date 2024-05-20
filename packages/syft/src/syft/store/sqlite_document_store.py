# future
from __future__ import annotations

# stdlib
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
import threading
from typing import Any

# third party
from pydantic import Field
from pydantic import field_validator
from typing_extensions import Self

# relative
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..types.uid import UID
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import LockingConfig
from .locks import NoLockingConfig


def _repr_debug_(value: Any) -> str:
    if hasattr(value, "_repr_debug_"):
        return str(value._repr_debug_())
    return repr(value)


@serializable(attrs=["index_name", "settings", "store_config"])
class SQLiteBackingStore(KeyValueBackingStore):
    """Core Store logic for the SQLite stores.

    Parameters:
        `index_name`: str
            Index name
        `settings`: PartitionSettings
            Syft specific settings
        `store_config`: SQLiteStoreConfig
            Connection Configuration
        `ddtype`: Type
            Class used as fallback on `get` errors
    """

    def __init__(
        self,
        index_name: str,
        settings: PartitionSettings,
        store_config: StoreConfig,
        ddtype: type | None = None,
    ) -> None:
        self.index_name = index_name
        self.settings = settings
        self.store_config = store_config
        self._ddtype = ddtype
        if self.store_config.client_config:
            self.file_path = self.store_config.client_config.file_path
        if store_config.client_config:
            self.db_filename = store_config.client_config.filename
        path = Path(self.file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.create_table()

    @property
    def table_name(self) -> str:
        return f"{self.settings.name}_{self.index_name}"

    def create_table(self) -> None:
        with self._cursor(
            f"create table if not exists {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, "  # nosec
            + "repr TEXT NOT NULL, value BLOB NOT NULL, "  # nosec
            + "sqltime TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)"
        ) as _:
            pass

    def _initialize_connection(self, con: sqlite3.Connection) -> None:
        """Set PRAGMA settings for the connection."""
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA busy_timeout = 5000")
        con.execute("PRAGMA temp_store = 2")
        con.execute("PRAGMA synchronous = 1")

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "connection"):
            timeout = (
                self.store_config.client_config.timeout
                if self.store_config.client_config
                else 5
            )
            self._local.connection = sqlite3.connect(
                self.file_path, timeout=timeout, check_same_thread=False
            )
            self._initialize_connection(self._local.connection)
        return self._local.connection

    @contextmanager
    def _cursor(
        self, sql: str, *args: list[Any] | None
    ) -> Generator[sqlite3.Cursor, None, None]:
        con = con = self._get_connection()
        cur = con.cursor()
        yield cur.execute(sql, *args)
        try:
            con.commit()
        finally:
            cur.close()

    def _set(self, key: UID, value: Any) -> None:
        if self._exists(key):
            self._update(key, value)
        else:
            insert_sql = (
                f"insert into {self.table_name} (uid, repr, value) VALUES (?, ?, ?)"  # nosec
            )
            data = _serialize(value, to_bytes=True)
            with self._cursor(insert_sql, [str(key), _repr_debug_(value), data]) as _:
                pass

    def _update(self, key: UID, value: Any) -> None:
        insert_sql = (
            f"update {self.table_name} set uid = ?, repr = ?, value = ? where uid = ?"  # nosec
        )
        data = _serialize(value, to_bytes=True)
        with self._cursor(
            insert_sql, [str(key), _repr_debug_(value), data, str(key)]
        ) as _:
            pass

    def _get(self, key: UID) -> Any:
        select_sql = f"select * from {self.table_name} where uid = ? order by sqltime"  # nosec
        with self._cursor(select_sql, [str(key)]) as cursor:
            row = cursor.fetchone()
            if row is None or len(row) == 0:
                raise KeyError(f"{key} not in {type(self)}")
            data = row[2]
            return _deserialize(data, from_bytes=True)

    def _exists(self, key: UID) -> bool:
        select_sql = f"select uid from {self.table_name} where uid = ?"  # nosec

        with self._cursor(select_sql, [str(key)]) as cursor:
            row = cursor.fetchone()
            return bool(row)

    def _get_all(self) -> Any:
        select_sql = f"select * from {self.table_name} order by sqltime"  # nosec
        keys = []
        data = []

        with self._cursor(select_sql) as cursor:
            rows = cursor.fetchall() or []

            for row in rows:
                keys.append(UID(row[0]))
                data.append(_deserialize(row[2], from_bytes=True))
            return dict(zip(keys, data))

    def _get_all_keys(self) -> Any:
        select_sql = f"select uid from {self.table_name} order by sqltime"  # nosec
        keys = []

        with self._cursor(select_sql) as cursor:
            rows = cursor.fetchall() or []

            for row in rows:
                keys.append(UID(row[0]))
            return keys

    def _delete(self, key: UID) -> None:
        select_sql = f"delete from {self.table_name} where uid = ?"  # nosec
        with self._cursor(select_sql, [str(key)]) as _:
            pass

    def _delete_all(self) -> None:
        select_sql = f"delete from {self.table_name}"  # nosec
        with self._cursor(select_sql) as _:
            pass

    def _len(self) -> int:
        select_sql = f"select count(uid) from {self.table_name}"  # nosec
        with self._cursor(select_sql) as cursor:
            cnt = cursor.fetchone()[0]
            return cnt

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

    def __delitem__(self, key: str) -> None:
        self._delete(key)

    def clear(self) -> None:
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

    def __del__(self) -> None:
        if hasattr(self._local, "connection"):
            self._local.connection.close()


@serializable()
class SQLiteStorePartition(KeyValueStorePartition):
    """SQLite StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for indexing and partitioning
        `store_config`: SQLiteStoreConfig
            SQLite specific configuration
    """


# the base document store is already a dict but we can change it later
@serializable()
class SQLiteDocumentStore(DocumentStore):
    """SQLite Document Store

    Parameters:
        `store_config`: StoreConfig
            SQLite specific configuration, including connection details and client class type.
    """

    partition_type = SQLiteStorePartition


@serializable()
class SQLiteStoreClientConfig(StoreClientConfig):
    """SQLite connection config

    Parameters:
        `filename` : str
            Database name
        `path` : Path or str
            Database folder
        `check_same_thread`: bool
            If True (default), ProgrammingError will be raised if the database connection is used
            by a thread other than the one that created it. If False, the connection may be accessed
            in multiple threads; write operations may need to be serialized by the user to avoid
            data corruption.
        `timeout`: int
            How many seconds the connection should wait before raising an exception, if the database
            is locked by another connection. If another connection opens a transaction to modify the
            database, it will be locked until that transaction is committed. Default five seconds.
    """

    filename: str = "syftdb.sqlite"
    path: str | Path = Field(default_factory=tempfile.gettempdir)
    check_same_thread: bool = True
    timeout: int = 5

    # We need this in addition to Field(default_factory=...)
    # so users can still do SQLiteStoreClientConfig(path=None)
    @field_validator("path", mode="before")
    @classmethod
    def __default_path(cls, path: str | Path | None) -> str | Path:
        if path is None:
            return tempfile.gettempdir()
        return path

    @property
    def file_path(self) -> Path | None:
        return Path(self.path) / self.filename


@serializable()
class SQLiteStoreConfig(StoreConfig):
    __canonical_name__ = "SQLiteStoreConfig"
    """SQLite Store config, used by SQLiteStorePartition

    Parameters:
        `client_config`: SQLiteStoreClientConfig
            SQLite connection configuration
        `store_type`: DocumentStore
            Class interacting with QueueStash. Default: SQLiteDocumentStore
        `backing_store`: KeyValueBackingStore
            The Store core logic. Default: SQLiteBackingStore
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
            Defaults to NoLockingConfig.
    """

    client_config: SQLiteStoreClientConfig
    store_type: type[DocumentStore] = SQLiteDocumentStore
    backing_store: type[KeyValueBackingStore] = SQLiteBackingStore
    locking_config: LockingConfig = Field(default_factory=NoLockingConfig)
