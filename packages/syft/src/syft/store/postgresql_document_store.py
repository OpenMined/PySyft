# stdlib
from collections import defaultdict
import logging

# third party
import psycopg2
from psycopg2.extensions import connection
from psycopg2.extensions import cursor
from pydantic import Field

# relative
from ..serde.serializable import serializable
from ..types.errors import SyftException
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
_CONNECTION_POOL_DB: dict[str, connection] = {}
_CONNECTION_POOL_CUR: dict[str, cursor] = {}
REF_COUNTS: dict[str, int] = defaultdict(int)


# https://www.psycopg.org/docs/module.html#psycopg2.connect
@serializable(canonical_name="PostgreSQLStoreClientConfig", version=1)
class PostgreSQLStoreClientConfig(StoreClientConfig):
    dbname: str
    username: str
    password: str
    host: str
    port: int


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
        self._ddtype = ddtype
        if self.store_config.client_config:
            self.dbname = self.store_config.client_config.dbname

        self.lock = SyftLock(NoLockingConfig())
        self.create_table()
        REF_COUNTS[cache_key(self.dbname)] += 1

    def _connect(self) -> None:
        if self.store_config.client_config:
            connection = psycopg2.connect(
                dbname=self.store_config.client_config.dbname,
                user=self.store_config.client_config.user,
                password=self.store_config.client_config.password,
                host=self.store_config.client_config.host,
                port=self.store_config.client_config.port,
            )

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
        except Exception as e:
            public_message = special_exception_public_message(self.table_name, e)
            raise SyftException.from_exception(e, public_message=public_message)

    @property
    def db(self) -> connection:
        if cache_key(self.dbname) not in _CONNECTION_POOL_DB:
            self._connect()
        return _CONNECTION_POOL_DB[cache_key(self.dbname)]

    @property
    def cur(self) -> cursor:
        if cache_key(self.db_filename) not in _CONNECTION_POOL_CUR:
            _CONNECTION_POOL_CUR[cache_key(self.dbname)] = self.db.cursor()

        return _CONNECTION_POOL_CUR[cache_key(self.dbname)]

    def _close(self) -> None:
        self._commit()
        REF_COUNTS[cache_key(self.db_filename)] -= 1
        if REF_COUNTS[cache_key(self.db_filename)] <= 0:
            # once you close it seems like other object references can't re-use the
            # same connection

            self.db.close()
            db_key = cache_key(self.db_filename)
            if db_key in _CONNECTION_POOL_CUR:
                # NOTE if we don't remove the cursor, the cursor cache_key can clash with a future thread id
                del _CONNECTION_POOL_CUR[db_key]
            del _CONNECTION_POOL_DB[cache_key(self.db_filename)]
        else:
            # don't close yet because another SQLiteBackingStore is probably still open
            pass


@serializable()
class PostgreSQLStoreConfig(StoreConfig):
    __canonical_name__ = "PostgreSQLStorePartition"

    client_config: PostgreSQLStoreClientConfig
    store_type: type[DocumentStore] = PostgreSQLDocumentStore
    backing_store: type[KeyValueBackingStore] = PostgreSQLBackingStore
    locking_config: LockingConfig = Field(default_factory=NoLockingConfig)
