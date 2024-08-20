# future
from __future__ import annotations

# stdlib
from typing import Any

# third party
from pydantic import Field

# relative
from ..serde.serializable import serializable
from ..server.credentials import SyftVerifyKey
from ..types import uid
from .document_store import DocumentStore
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import LockingConfig
from .locks import ThreadingLockingConfig


@serializable(canonical_name="DictBackingStore", version=1)
class DictBackingStore(dict, KeyValueBackingStore):  # type: ignore[misc]
    """Dictionary-based Store core logic

    This class provides the core logic for a dictionary-based key-value store.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the dictionary-based backing store.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.
        """
        super().__init__()
        self._ddtype = kwargs.get("ddtype", None)

    def __getitem__(self, key: Any) -> Any:
        """Retrieve an item from the store by key.

        Args:
            key (Any): The key of the item to retrieve.

        Returns:
            Any: The value associated with the key.

        Raises:
            KeyError: If the key is not found in the store.
        """
        try:
            value = super().__getitem__(key)
            return value
        except KeyError as e:
            if self._ddtype:
                return self._ddtype()
            raise e


@serializable(canonical_name="DictStorePartition", version=1)
class DictStorePartition(KeyValueStorePartition):
    """Dictionary-based StorePartition

    This class represents a partition within a dictionary-based key-value store.

    Parameters:
        settings (PartitionSettings): PySyft specific settings, used for indexing and partitioning.
        store_config (DictStoreConfig): Dictionary Store specific configuration.
    """

    def prune(self) -> None:
        """Reset the partition by reinitializing the store."""
        self.init_store()


@serializable(canonical_name="DictDocumentStore", version=1)
class DictDocumentStore(DocumentStore):
    """Dictionary-based Document Store

    This class represents a document store implemented using a dictionary.
    """

    partition_type = DictStorePartition

    def __init__(
        self,
        server_uid: uid,
        root_verify_key: SyftVerifyKey | None,
        store_config: DictStoreConfig | None = None,
    ) -> None:
        if store_config is None:
            store_config = DictStoreConfig()
        super().__init__(
            server_uid=server_uid,
            root_verify_key=root_verify_key,
            store_config=store_config,
        )

    def reset(self) -> None:
        """Reset the document store by pruning all partitions."""
        for partition in self.partitions.values():
            partition.prune()


@serializable()
class DictStoreConfig(StoreConfig):
    """Dictionary-based configuration

    This class provides the configuration for a dictionary-based document store.

    Attributes:
        store_type (type[DocumentStore]): The Document type used. Default: DictDocumentStore.
        backing_store (type[KeyValueBackingStore]): The backend type used. Default: DictBackingStore.
        locking_config (LockingConfig): The config used for store locking. Defaults to ThreadingLockingConfig.

    Parameters:
        store_type (Type[DocumentStore]): The Document type used. Default: DictDocumentStore.
        backing_store (Type[KeyValueBackingStore]): The backend type used. Default: DictBackingStore.
        locking_config (LockingConfig): The config used for store locking.
    """

    store_type: type[DocumentStore] = DictDocumentStore
    backing_store: type[KeyValueBackingStore] = DictBackingStore
    locking_config: LockingConfig = Field(default_factory=ThreadingLockingConfig)
