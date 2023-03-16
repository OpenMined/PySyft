# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Type

# relative
from .document_store import DocumentStore
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .serializable import serializable


@serializable(recursive_serde=True)
class DictBackingStore(dict, KeyValueBackingStore):
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


@serializable(recursive_serde=True)
class DictStorePartition(KeyValueStorePartition):
    def prune(self):
        self.init_store()


# the base document store is already a dict but we can change it later
@serializable(recursive_serde=True)
class DictDocumentStore(DocumentStore):
    partition_type = DictStorePartition

    def __init__(self, store_config: Optional[DictStoreConfig] = None) -> None:
        if store_config is None:
            store_config = DictStoreConfig()
        super().__init__(store_config=store_config)

    def reset(self):
        for _, partition in self.partitions.items():
            partition.prune()


@serializable(recursive_serde=True)
class DictStoreConfig(StoreConfig):
    store_type: Type[DocumentStore] = DictDocumentStore
    backing_store: Type[KeyValueBackingStore] = DictBackingStore
