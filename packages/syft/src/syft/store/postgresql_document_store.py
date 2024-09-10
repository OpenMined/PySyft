# stdlib
import logging
from typing import Any

# third party
import psycopg
from psycopg import Connection
from psycopg import Cursor
from psycopg.errors import DuplicateTable
from psycopg.errors import InFailedSqlTransaction
from pydantic import Field
from typing_extensions import Self

# relative
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..types.errors import SyftException
from ..types.result import as_result
from ..types.uid import UID
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
from .sqlite_document_store import _repr_debug_
from .sqlite_document_store import cache_key
from .sqlite_document_store import special_exception_public_message

logger = logging.getLogger(__name__)
_CONNECTION_POOL_DB: dict[str, Connection] = {}


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
            _CONNECTION_POOL_DB[cache_key(self.dbname)] = connection
            print(f"Connected to {self.store_config.client_config.dbname}")
            print(
                "PostgreSQL database connection:",
                _CONNECTION_POOL_DB[cache_key(self.dbname)],
            )

    def create_table(self) -> None:
        db = self.db
        try:
            with self.lock:
                with db.cursor() as cur:
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, "  # nosec
                        + "repr TEXT NOT NULL, value BYTEA NOT NULL, "  # nosec
                        + "sqltime TIMESTAMP NOT NULL DEFAULT NOW())"  # nosec
                    )
                    cur.connection.commit()
        except DuplicateTable:
            pass
        except InFailedSqlTransaction:
            db.rollback()
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
        return self.db.cursor()

    @staticmethod
    @as_result(SyftException)
    def _execute(
        lock: SyftLock,
        cursor: Cursor,
        db: Connection,
        table_name: str,
        sql: str,
        args: list[Any] | None,
    ) -> Cursor:
        with lock:
            try:
                cursor.execute(sql, args)  # Execute the SQL with arguments
                db.commit()  # Commit if everything went ok
            except InFailedSqlTransaction as ie:
                db.rollback()  # Rollback if something went wrong
                raise SyftException(
                    public_message=f"Transaction `{sql}` failed and was rolled back. \n"
                    f"Error: {ie}."
                )
            except Exception as e:
                logger.debug(f"Rolling back SQL: {sql} with args: {args}")
                db.rollback()  # Rollback on any other exception to maintain clean state
                public_message = special_exception_public_message(table_name, e)
                logger.error(public_message)
                raise SyftException.from_exception(e, public_message=public_message)
        return cursor

    def _set(self, key: UID, value: Any) -> None:
        if self._exists(key):
            self._update(key, value)
        else:
            insert_sql = (
                f"insert into {self.table_name} (uid, repr, value) VALUES "  # nosec
                f"({self.subs_char}, {self.subs_char}, {self.subs_char})"  # nosec
            )
            data = _serialize(value, to_bytes=True)
            with self.cur as cur:
                self._execute(
                    self.lock,
                    cur,
                    cur.connection,
                    self.table_name,
                    insert_sql,
                    [str(key), _repr_debug_(value), data],
                ).unwrap()

    def _update(self, key: UID, value: Any) -> None:
        insert_sql = (
            f"update {self.table_name} set uid = {self.subs_char}, "  # nosec
            f"repr = {self.subs_char}, value = {self.subs_char} "  # nosec
            f"where uid = {self.subs_char}"  # nosec
        )
        data = _serialize(value, to_bytes=True)
        with self.cur as cur:
            self._execute(
                self.lock,
                cur,
                cur.connection,
                self.table_name,
                insert_sql,
                [str(key), _repr_debug_(value), data, str(key)],
            ).unwrap()

    def _get(self, key: UID) -> Any:
        select_sql = (
            f"select * from {self.table_name} where uid = {self.subs_char} "  # nosec
            "order by sqltime"
        )
        with self.cur as cur:
            cursor = self._execute(
                self.lock, cur, cur.connection, self.table_name, select_sql, [str(key)]
            ).unwrap(public_message=f"Query {select_sql} failed")
            row = cursor.fetchone()
        if row is None or len(row) == 0:
            raise KeyError(f"{key} not in {type(self)}")
        data = row[2]
        return _deserialize(data, from_bytes=True)

    def _exists(self, key: UID) -> bool:
        select_sql = f"select uid from {self.table_name} where uid = {self.subs_char}"  # nosec
        row = None
        with self.cur as cur:
            cursor = self._execute(
                self.lock, cur, cur.connection, self.table_name, select_sql, [str(key)]
            ).unwrap()
            row = cursor.fetchone()  # type: ignore
        if row is None:
            return False
        return bool(row)

    def _get_all(self) -> Any:
        select_sql = f"select * from {self.table_name} order by sqltime"  # nosec
        keys = []
        data = []
        with self.cur as cur:
            cursor = self._execute(
                self.lock, cur, cur.connection, self.table_name, select_sql, []
            ).unwrap()
            rows = cursor.fetchall()  # type: ignore
            if not rows:
                return {}

        for row in rows:
            keys.append(UID(row[0]))
            data.append(_deserialize(row[2], from_bytes=True))

        return dict(zip(keys, data))

    def _get_all_keys(self) -> Any:
        select_sql = f"select uid from {self.table_name} order by sqltime"  # nosec
        with self.cur as cur:
            cursor = self._execute(
                self.lock, cur, cur.connection, self.table_name, select_sql, []
            ).unwrap()
            rows = cursor.fetchall()  # type: ignore
        if not rows:
            return []
        keys = [UID(row[0]) for row in rows]
        return keys

    def _delete(self, key: UID) -> None:
        select_sql = f"delete from {self.table_name} where uid = {self.subs_char}"  # nosec
        with self.cur as cur:
            self._execute(
                self.lock, cur, self.table_name, select_sql, [str(key)]
            ).unwrap()

    def _delete_all(self) -> None:
        select_sql = f"delete from {self.table_name}"  # nosec
        with self.cur as cur:
            self._execute(
                self.lock, cur, cur.connection, self.table_name, select_sql, []
            ).unwrap()

    def _len(self) -> int:
        select_sql = f"select count(uid) from {self.table_name}"  # nosec
        with self.cur as cur:
            cursor = self._execute(
                self.lock, cur, cur.connection, self.table_name, select_sql, []
            ).unwrap()
            cnt = cursor.fetchone()[0]
        return cnt

    def _close(self) -> None:
        self._commit()
        if cache_key(self.dbname) in _CONNECTION_POOL_DB:
            conn = _CONNECTION_POOL_DB[cache_key(self.dbname)]
            conn.close()
            _CONNECTION_POOL_DB.pop(cache_key(self.dbname), None)

    def _commit(self) -> None:
        self.db.commit()


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
