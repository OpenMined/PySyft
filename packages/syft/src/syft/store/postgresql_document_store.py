# stdlib
from collections import defaultdict
import logging
from typing import Any
from typing import Self

# third party
import psycopg
from psycopg import Connection
from psycopg import Cursor
from psycopg.errors import DuplicateTable
from psycopg.errors import InFailedSqlTransaction
from pydantic import Field

# relative
from ..serde.serializable import serializable
from ..types.errors import SyftException
from ..types.result import as_result
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .locks import LockingConfig
from .locks import NoLockingConfig
from .locks import SyftLock
from .sqlite_document_store import SQLiteBackingStore
from .sqlite_document_store import SQLiteStorePartition
from .sqlite_document_store import cache_key
from .sqlite_document_store import special_exception_public_message

logger = logging.getLogger(__name__)
_CONNECTION_POOL_DB: dict[str, Connection] = {}
_CONNECTION_POOL_CUR: dict[str, Cursor] = {}
REF_COUNTS: dict[str, int] = defaultdict(int)


# https://www.psycopg.org/docs/module.html#psycopg2.connect
@serializable(canonical_name="PostgreSQLStoreClientConfig", version=1)
class PostgreSQLStoreClientConfig(StoreClientConfig):
    dbname: str
    username: str
    password: str
    host: str
    port: int

    # makes hashabel
    class Config:
        frozen = True

    def __hash__(self) -> int:
        return hash((self.dbname, self.username, self.password, self.host, self.port))

    def __str__(self) -> str:
        return f"dbname={self.dbname} user={self.username} password={self.password} host={self.host} port={self.port}"


@serializable(canonical_name="PostgreSQLStorePartition", version=1)
class PostgreSQLStorePartition(SQLiteStorePartition):
    pass


@serializable(canonical_name="PostgreSQLDocumentStore", version=1)
class PostgreSQLDocumentStore(DocumentStore):
    partition_type = PostgreSQLStorePartition


@serializable(
    attrs=["index_name", "settings", "store_config"],
    canonical_name="PostgreSQLBackingStore",
    version=1,
)
class PostgreSQLBackingStore(SQLiteBackingStore):
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
        self.store_config_hash = hash(store_config.client_config)
        self._ddtype = ddtype
        if self.store_config.client_config:
            self.dbname = self.store_config.client_config.dbname

        self.lock = SyftLock(NoLockingConfig())
        self.create_table()
        REF_COUNTS[cache_key(self.dbname)] += 1
        self.subs_char = r"%s"  # thanks postgresql

    def _connect(self) -> None:
        if self.store_config.client_config:
            connection = psycopg.connect(
                dbname=self.store_config.client_config.dbname,
                user=self.store_config.client_config.username,
                password=self.store_config.client_config.password,
                host=self.store_config.client_config.host,
                port=self.store_config.client_config.port,
            )

            print(f"Connected to {self.store_config.client_config.dbname}")
            print("PostgreSQL database connection:", connection._check_connection_ok())

            _CONNECTION_POOL_DB[cache_key(self.dbname)] = connection

    def create_table(self) -> None:
        try:
            with self.lock:
                self.cur.execute(
                    f"create table {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, "  # nosec
                    + "repr TEXT NOT NULL, value BYTEA NOT NULL, "  # nosec
                    + "sqltime TIMESTAMP NOT NULL DEFAULT NOW())"  # nosec
                )
                self.db.commit()
        except DuplicateTable:
            pass
        except InFailedSqlTransaction:
            self.db.rollback()
        except Exception as e:
            public_message = special_exception_public_message(self.table_name, e)
            raise SyftException.from_exception(e, public_message=public_message)

    @property
    def db(self) -> Connection:
        if cache_key(self.dbname) not in _CONNECTION_POOL_DB:
            self._connect()
        return _CONNECTION_POOL_DB[cache_key(self.dbname)]

    @property
    def cur(self) -> Cursor:
        if cache_key(self.store_config_hash) not in _CONNECTION_POOL_CUR:
            _CONNECTION_POOL_CUR[cache_key(self.dbname)] = self.db.cursor()

        return _CONNECTION_POOL_CUR[cache_key(self.dbname)]

    def _close(self) -> None:
        self._commit()
        REF_COUNTS[cache_key(self.store_config_hash)] -= 1
        if REF_COUNTS[cache_key(self.store_config_hash)] <= 0:
            # once you close it seems like other object references can't re-use the
            # same connection

            self.db.close()
            db_key = cache_key(self.store_config_hash)
            if db_key in _CONNECTION_POOL_CUR:
                # NOTE if we don't remove the cursor, the cursor cache_key can clash with a future thread id
                del _CONNECTION_POOL_CUR[db_key]
            del _CONNECTION_POOL_DB[cache_key(self.store_config_hash)]
        else:
            # don't close yet because another SQLiteBackingStore is probably still open
            pass

    @as_result(SyftException)
    def _execute(self, sql: str, args: list[Any] | None) -> psycopg.Cursor:
        with self.lock:
            cursor: psycopg.Cursor | None = None
            try:
                # Ensure self.cur is a psycopg cursor object
                cursor = self.cur  # Assuming self.cur is already set as psycopg.Cursor
                cursor.execute(sql, args)  # Execute the SQL with arguments
                # cursor = self.cur.execute(sql, args)
            except InFailedSqlTransaction:
                self.db.rollback()  # Rollback if something went wrong
                raise SyftException(
                    public_message=f"Transaction {sql} failed and was rolled back."
                )
            except Exception as e:
                self.db.rollback()  # Rollback on any other exception to maintain clean state
                public_message = special_exception_public_message(self.table_name, e)
                raise SyftException.from_exception(e, public_message=public_message)
            self.db.commit()  # Commit if everything went ok
            return cursor


@serializable()
class PostgreSQLStoreConfig(StoreConfig):
    __canonical_name__ = "PostgreSQLStorePartition"

    client_config: PostgreSQLStoreClientConfig
    store_type: type[DocumentStore] = PostgreSQLDocumentStore
    backing_store: type[KeyValueBackingStore] = PostgreSQLBackingStore
    locking_config: LockingConfig = Field(default_factory=NoLockingConfig)

    @classmethod
    def from_dict(cls, client_config_dict: dict) -> Self:
        postgresql_client_config = PostgreSQLStoreClientConfig(
            dbname=client_config_dict["POSTGRESQL_DBNAME"],
            host=client_config_dict["POSTGRESQL_HOST"],
            port=client_config_dict["POSTGRESQL_PORT"],
            username=client_config_dict["POSTGRESQL_USERNAME"],
            password=client_config_dict["POSTGRESQL_PASSWORD"],
        )

        return PostgreSQLStoreConfig(client_config=postgresql_client_config)
