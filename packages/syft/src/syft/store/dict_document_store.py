# future
from __future__ import annotations

# stdlib
from typing import Any

# third party
from pydantic import Field

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.serializable import serializable
from ..types import uid
from .document_store import DocumentStore
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import LockingConfig
from .locks import ThreadingLockingConfig


@serializable()
class DictBackingStore(dict, KeyValueBackingStore):  # type: ignore[misc]
    # TODO: fix the mypy issue
    """Dictionary-based Store core logic"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
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

    def prune(self) -> None:
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
        node_uid: uid,
        root_verify_key: SyftVerifyKey | None,
        store_config: DictStoreConfig | None = None,
    ) -> None:
        if store_config is None:
            store_config = DictStoreConfig()
        super().__init__(
            node_uid=node_uid,
            root_verify_key=root_verify_key,
            store_config=store_config,
        )

    def reset(self) -> None:
        for _, partition in self.partitions.items():
            partition.prune()


@serializable()
class DictStoreConfig(StoreConfig):
    __canonical_name__ = "DictStoreConfig"
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
            Defaults to ThreadingLockingConfig.
    """

    store_type: type[DocumentStore] = DictDocumentStore
    backing_store: type[KeyValueBackingStore] = DictBackingStore
    locking_config: LockingConfig = Field(default_factory=ThreadingLockingConfig)
