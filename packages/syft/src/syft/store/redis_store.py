# future
from __future__ import annotations

# stdlib
from copy import deepcopy
import sqlite3
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
import OpenSSL
import redis
from redis import ConnectionPool
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..types.uid import UID
from ..util.util import thread_ident
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import FileLockingConfig
from .locks import LockingConfig


def _repr_debug_(value: Any) -> str:
    if hasattr(value, "_repr_debug_"):
        return str(value._repr_debug_())
    return repr(value)


@serializable(attrs=["index_name", "settings", "store_config"])
class RedisBackingStore(KeyValueBackingStore):
    """Core Store logic for the Redis stores.

    Parameters:
        `index_name`: str
            Index name
        `settings`: PartitionSettings
            Syft specific settings
        `store_config`: RedisStoreConfig
            Connection Configuration
        `ddtype`: Type
            Class used as fallback on `get` errors
    """

    def __init__(
        self,
        index_name: str,
        settings: PartitionSettings,
        store_config: StoreConfig,
        ddtype: Optional[type] = None,
    ) -> None:
        self.index_name = index_name
        self.settings = settings
        self.store_config = store_config
        self._ddtype = ddtype
        self._db: Dict[int, sqlite3.Connection] = {}
        # self._cur: Dict[int, sqlite3.Cursor] = {}

    @property
    def db_name(self) -> str:
        return f"{self.settings.name}_{self.index_name}"

    def _connect(self) -> None:
        # SQLite is not thread safe by default so we ensure that each connection
        # comes from a different thread. In cases of Uvicorn and other AWSGI servers
        # there will be many threads handling incoming requests so we need to ensure
        # that different connections are used in each thread. By using a dict for the
        # _db and _cur we can ensure they are never shared
        redis_config: RedisStoreClientConfig = self.store_config.client_config
        self._db[thread_ident()] = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=self.db_name,
            username=redis_config.username,
            password=redis_config.password,
            client_name=redis_config.client_name,
            socket_timeout=redis_config.socket_timeout,
            socket_connect_timeout=redis_config.socket_connect_timeout,
            socket_keepalive=redis_config.socket_keepalive,
            retry=redis_config.retry,
            max_connections=redis_config.max_connections,
            connection_pool=redis_config.connection_pool,
            health_check_interval=redis_config.health_check_interval,
            encoding=redis_config.encoding,
            ssl=redis_config.ssl,
            ssl_keyfile=redis_config.ssl_keyfile,
            ssl_certfile=redis_config.ssl_certfile,
            ssl_cert_reqs=redis_config.ssl_cert_reqs,
            ssl_ca_certs=redis_config.ssl_ca_certs,
            ssl_ca_path=redis_config.ssl_ca_path,
            ssl_ca_data=redis_config.ssl_ca_data,
            ssl_check_hostname=redis_config.ssl_check_hostname,
            ssl_password=redis_config.ssl_password,
            ssl_validate_ocsp=redis_config.ssl_validate_ocsp,
            ssl_validate_ocsp_stapled=redis_config.ssl_validate_ocsp_stapled,
            ssl_ocsp_context=redis_config.ssl_ocsp_context,
            ssl_ocsp_expected_cert=redis_config.ssl_ocsp_expected_cert,
            single_connection_client=redis_config.single_connection_client,
        )

    def create_table(self):
        try:
            self.cur.execute(
                f"create table {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, "  # nosec
                + "repr TEXT NOT NULL, value BLOB NOT NULL, "  # nosec
                + "sqltime TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)"  # nosec
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

    def _execute(
        self, sql: str, *args: Optional[List[Any]]
    ) -> Result[Ok[sqlite3.Cursor], Err[str]]:
        pass

    def _update(self, key: UID, value: Any) -> None:
        pass

    def _get(self, key: UID) -> Any:
        pass

    def _exists(self, key: UID) -> bool:
        pass

    def _get_all(self) -> Any:
        pass

    def _get_all_keys(self) -> Any:
        pass

    def _delete(self, key: UID) -> None:
        pass

    def _delete_all(self) -> None:
        pass

    def _len(self) -> int:
        pass

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
        # NOTE: not thread-safe
        value = self._get(key)
        self._delete(key)
        return value

    def __contains__(self, key: Any) -> bool:
        return self._exists(key)

    def __iter__(self) -> Any:
        return iter(self.keys())

    def __del__(self):
        try:
            self._close()
        except BaseException:
            pass


@serializable()
class RedisStorePartition(KeyValueStorePartition):
    """Redis StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for indexing and partitioning
        `store_config`: SQLiteStoreConfig
            SQLite specific configuration
    """

    def close(self) -> None:
        self.lock.acquire()
        try:
            self.data._close()
            self.unique_keys._close()
            self.searchable_keys._close()
        except BaseException:
            pass
        self.lock.release()

    def commit(self) -> None:
        self.lock.acquire()
        try:
            self.data._commit()
            self.unique_keys._commit()
            self.searchable_keys._commit()
        except BaseException:
            pass
        self.lock.release()


# the base document store is already a dict but we can change it later
@serializable()
class RedisDocumentStore(DocumentStore):
    """Redis Document Store

    Parameters:
        `store_config`: StoreConfig
            Redis specific configuration, including connection details and client class type.
    """

    partition_type = RedisStorePartition


@serializable()
class RedisStoreClientConfig(StoreClientConfig):
    """Redis connection config

    Parameters:
        `single_connection_client`: bool
            if `True`, connection pool is not used. In that case `Redis`
            instance use is not thread safe.
        `ssl`:
            If `True`, ssl settings are used to create a connection. Defaults to `False`.
        `ssl_keyfile`:
            Path to an ssl private key. Defaults to None.
        `ssl_certfile`:
            Path to an ssl certificate. Defaults to None.
        `ssl_cert_reqs`:
            The string value for the SSLContext.verify_mode (none, optional, required).
            Defaults to "required".
        `ssl_ca_certs`:
            The path to a file of concatenated CA certificates in PEM format.
            Defaults to None.
        `ssl_ca_data`:
            Either an ASCII string of one or more PEM-encoded certificates
            or a bytes-like object of DER-encoded certificates.
        `ssl_check_hostname`:
            If set, match the hostname during the SSL handshake.
            Defaults to False.
        `ssl_ca_path`:
            The path to a directory containing several CA certificates in PEM format.
            Defaults to None.
        `ssl_password`:
            Password for unlocking an encrypted private key. Defaults to None.
        `ssl_validate_ocsp`:
            If set, perform a full ocsp validation (i.e not a stapled verification)
        `ssl_validate_ocsp_stapled`:
            If set, perform a validation on a stapled ocsp response
        `ssl_ocsp_context`:
            A fully initialized OpenSSL.SSL.Context object to be used in
            verifying the ssl_ocsp_expected_cert
        `ssl_ocsp_expected_cert`:
            A PEM armoured string containing the expected certificate to be
            returned from the ocsp verification service.
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: Optional[str]
    password: Optional[str]
    client_name: Optional[str]
    socket_timeout: Optional[int]
    socket_connect_timeout: Optional[int]
    socket_keepalive: Optional[bool]
    retry: Optional[int]
    max_connections: Optional[int]
    connection_pool: Optional[ConnectionPool]
    health_check_interval: int = 0
    encoding: str = "utf-8"
    ssl: bool = False
    ssl_keyfile: Optional[str]
    ssl_certfile: Optional[str]
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str]
    ssl_ca_path: Optional[str]
    ssl_ca_data: Optional[bytes]
    ssl_check_hostname: Optional[False]
    ssl_password: Optional[str]
    ssl_validate_ocsp: bool = False
    ssl_validate_ocsp_stapled: bool = False
    ssl_ocsp_context: Optional[OpenSSL.SSL.Context]
    ssl_ocsp_expected_cert: Optional[bytes]
    single_connection_client: bool = False


@serializable()
class RedisStoreConfig(StoreConfig):
    """SQLite Store config, used by SQLiteStorePartition

    Parameters:
        `client_config`: RedisStoreClientConfig
            SQLite connection configuration
        `store_type`: DocumentStore
            Class interacting with QueueStash. Default: SQLiteDocumentStore
        `backing_store`: RedisBackingStore
            The Store core logic. Default: SQLiteBackingStore
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
                * FileLockingConfig: file based locking, ideal for same-device different-processes/threads stores.
                * RedisLockingConfig: Redis-based locking, ideal for multi-device stores.
            Defaults to FileLockingConfig.
    """

    client_config: RedisStoreClientConfig
    store_type: Type[DocumentStore] = RedisDocumentStore
    backing_store: Type[KeyValueBackingStore] = RedisBackingStore
    locking_config: LockingConfig = FileLockingConfig()
