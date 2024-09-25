# third party
from pydantic import Field

# relative
from ..serde.serializable import serializable
from .document_store import DocumentStore
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import LockingConfig
from .locks import NoLockingConfig


@serializable(
    attrs=["index_name", "settings", "store_config"],
    canonical_name="SQLiteBackingStore",
    version=1,
)
class SQLiteBackingStore(KeyValueBackingStore):
    """Core Store logic for the SQLite stores."""

    pass


@serializable(canonical_name="SQLiteStorePartition", version=1)
class SQLiteStorePartition(KeyValueStorePartition):
    """SQLite StorePartition
    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for indexing and partitioning
        `store_config`: SQLiteStoreConfig
            SQLite specific configuration
    """


# the base document store is already a dict but we can change it later
@serializable(canonical_name="SQLiteDocumentStore", version=1)
class SQLiteDocumentStore(DocumentStore):
    """SQLite Document Store
    Parameters:
        `store_config`: StoreConfig
            SQLite specific configuration, including connection details and client class type.
    """

    partition_type = SQLiteStorePartition


@serializable(canonical_name="SQLiteStoreClientConfig", version=1)
class SQLiteStoreClientConfig(StoreClientConfig):
    """SQLite connection config"""

    pass


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
