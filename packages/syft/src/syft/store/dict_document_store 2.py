# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Type

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.serializable import serializable
from .document_store import DocumentStore
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import LockingConfig
from .locks import ThreadingLockingConfig


@serializable()
class DictBackingStore(dict, KeyValueBackingStore):
    """Dictionary-based Store core logic"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(dict).__init__()
        self._ddtype = kwargs.get("ddtype", None)

    def __getitem__(self, key: Any) -> Any:
        try:
            value = super().__getitem__(key)
            return value
        except KeyError as e:
            if self._ddtype:
                return self._ddtype()
            raise e


@serializable()
class DictStorePartition(KeyValueStorePartition):
    """Dictionary-based StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for indexing and partitioning
        `store_config`: DictStoreConfig
            DictStore specific configuration
    """

    def prune(self):
        self.init_store()


# the base document store is already a dict but we can change it later
@serializable()
class DictDocumentStore(DocumentStore):
    """Dictionary-based Document Store

    Parameters:
        `store_config`: DictStoreConfig
            Dictionary Store specific configuration, containing the store type and the backing store type
    """

    partition_type = DictStorePartition

    def __init__(
        self,
        root_verify_key: Optional[SyftVerifyKey],
        store_config: Optional[DictStoreConfig] = None,
    ) -> None:
        if store_config is None:
            store_config = DictStoreConfig()
        super().__init__(root_verify_key=root_verify_key, store_config=store_config)

    def reset(self):
        for _, partition in self.partitions.items():
            partition.prune()


@serializable()
class DictStoreConfig(StoreConfig):
    """Dictionary-based configuration

    Parameters:
        `store_type`: Type[DocumentStore]
            The Document type used. Default: DictDocumentStore
        `backing_store`: Type[KeyValueBackingStore]
            The backend type used. Default: DictBackingStore
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
                * FileLockingConfig: file based locking, ideal for same-device different-processes/threads stores.
                * RedisLockingConfig: Redis-based locking, ideal for multi-device stores.
            Defaults to ThreadingLockingConfig.
    """

    store_type: Type[DocumentStore] = DictDocumentStore
    backing_store: Type[KeyValueBackingStore] = DictBackingStore
    locking_config: LockingConfig = ThreadingLockingConfig()
