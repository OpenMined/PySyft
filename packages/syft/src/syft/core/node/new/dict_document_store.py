# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Type

# relative
from ...common.serde.serializable import serializable
from .document_store import DocumentStore
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition


class DictBackingStore(dict, KeyValueBackingStore):
    def __init__(self, *args: Any, **kwargs: Any) -> Dict:
        super(dict).__init__()


@serializable(recursive_serde=True)
class DictStorePartition(KeyValueStorePartition):
    pass


# the base document store is already a dict but we can change it later
@serializable(recursive_serde=True)
class DictDocumentStore(DocumentStore):
    partition_type = DictStorePartition


@serializable(recursive_serde=True)
class DictStoreConfig(StoreConfig):
    store_type: Type[DocumentStore] = DictDocumentStore
    backing_store: Type[KeyValueBackingStore] = DictBackingStore
