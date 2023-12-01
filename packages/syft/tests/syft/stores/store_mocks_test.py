# stdlib
from typing import Any
from typing import Type

# syft absolute
from syft.serde.serializable import serializable
from syft.store.document_store import DocumentStore
from syft.store.document_store import PartitionSettings
from syft.store.document_store import StoreConfig
from syft.store.kv_document_store import KeyValueBackingStore
from syft.types.syft_object import SyftObject
from syft.types.uid import UID


@serializable()
class MockKeyValueBackingStore(dict, KeyValueBackingStore):
    def __init__(
        self,
        index_name: str,
        settings: PartitionSettings,
        store_config: StoreConfig,
        **kwargs: Any,
    ) -> None:
        super(dict).__init__()
        self._ddtype = kwargs.get("ddtype", None)
        self.is_crashed = store_config.is_crashed

    def _check_if_crashed(self) -> None:
        # if self.is_crashed:
        #     raise RuntimeError("The backend is down")
        pass

    def __setitem__(self, key: Any, value: Any) -> None:
        self._check_if_crashed()
        value = super().__setitem__(key, value)
        return value

    def __getitem__(self, key: Any) -> Any:
        try:
            self._check_if_crashed()
            value = super().__getitem__(key)
            return value
        except KeyError as e:
            if self._ddtype:
                return self._ddtype()
            raise e


@serializable()
class MockObjectType(SyftObject):
    __canonical_name__ = "mock_type"


@serializable()
class MockStore(DocumentStore):
    __canonical_name__ = "MockStore"
    pass


@serializable()
class MockSyftObject(SyftObject):
    __canonical_name__ = str(UID())
    data: Any


@serializable()
class MockStoreConfig(StoreConfig):
    __canonical_name__ = "MockStoreConfig"
    store_type: Type[DocumentStore] = MockStore
    db_name: str = "testing"
    backing_store: Type[KeyValueBackingStore] = MockKeyValueBackingStore
    is_crashed: bool = False
