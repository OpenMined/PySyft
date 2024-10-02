# third party
from pydantic import Field

# relative
from ..serde.serializable import serializable
from ..types.syft_object import StorableObjectType
from .document_store import DocumentStore
from .document_store import StoreConfig
from .document_store import StorePartition
from .kv_document_store import KeyValueBackingStore
from .locks import LockingConfig
from .locks import NoLockingConfig
from .mongo_client import MongoStoreClientConfig


class MongoBsonObject(StorableObjectType):
    pass


@serializable(
    attrs=["index_name", "settings", "store_config"],
    canonical_name="MongoBackingStore",
    version=1,
)
class MongoBackingStore(KeyValueBackingStore):
    pass


@serializable(attrs=["storage_type"], canonical_name="MongoStorePartition", version=1)
class MongoStorePartition(StorePartition):
    """Mongo StorePartition
    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for partitioning and indexing.
        `store_config`: MongoStoreConfig
            Mongo specific configuration
    """

    storage_type: type[StorableObjectType] = MongoBsonObject


@serializable(canonical_name="MongoDocumentStore", version=1)
class MongoDocumentStore(DocumentStore):
    """Mongo Document Store
    Parameters:
        `store_config`: MongoStoreConfig
            Mongo specific configuration, including connection configuration, database name, or client class type.
    """

    partition_type = MongoStorePartition


@serializable()
class MongoStoreConfig(StoreConfig):
    __canonical_name__ = "MongoStoreConfig"

    """Mongo Store configuration
    Parameters:
        `client_config`: MongoStoreClientConfig
            Mongo connection details: hostname, port, user, password etc.
        `store_type`: Type[DocumentStore]
            The type of the DocumentStore. Default: MongoDocumentStore
        `db_name`: str
            Database name
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
            Defaults to NoLockingConfig.
    """

    client_config: MongoStoreClientConfig
    store_type: type[DocumentStore] = MongoDocumentStore
    db_name: str = "app"
    backing_store: type[KeyValueBackingStore] = MongoBackingStore
    # TODO: should use a distributed lock, with RedisLockingConfig
    locking_config: LockingConfig = Field(default_factory=NoLockingConfig)
