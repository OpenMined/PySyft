# stdlib
import copy
from typing import Any
from typing import Optional
from typing import Type

# third party
from typing_extensions import Self

# syft absolute
from syft.core.common.serde.serializable import serializable
from syft.core.common.uid import UID
from syft.core.node.common.node_table.syft_object import SyftObject
from syft.core.node.new.document_store import DocumentStore
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import StoreConfig
from syft.core.node.new.kv_document_store import KeyValueBackingStore


class MockKeyValueBackingStore(KeyValueBackingStore):
    def __init__(
        self,
        index_name: str,
        settings: PartitionSettings,
        store_config: StoreConfig,
        ddtype: Optional[type] = None,
    ) -> None:
        self.data = {}
        self.is_crashed = store_config.is_crashed

    def _check_if_crashed(self) -> None:
        if self.is_crashed:
            raise RuntimeError("The backend is down")

    def __setitem__(self, key: Any, value: Any) -> None:
        self._check_if_crashed()
        self.data[key] = copy.deepcopy(value)

    def __getitem__(self, key: Any) -> Self:
        self._check_if_crashed()
        return self.data[key]

    def __repr__(self) -> str:
        return "mock_store"

    def __len__(self) -> int:
        self._check_if_crashed()
        return len(self.data)

    def __delitem__(self, key: str):
        self._check_if_crashed()
        del self.data[key]

    def clear(self) -> Self:
        self._check_if_crashed()
        self.data = {}

    def copy(self) -> Self:
        self._check_if_crashed()
        return copy.deepcopy(self)

    def update(self, key: Any, value: Any) -> Self:
        self._check_if_crashed()

        self.data[key] = copy.deepcopy(value)

    def keys(self) -> Any:
        self._check_if_crashed()
        return self.data.keys()

    def values(self) -> Any:
        self._check_if_crashed()
        return self.data.values()

    def items(self) -> Any:
        self._check_if_crashed()
        return self.data.items()

    def pop(self, key: Any) -> Self:
        self._check_if_crashed()
        val = self.data[key]
        del self.data[key]
        return val

    def __contains__(self, key: Any) -> bool:
        self._check_if_crashed()
        return key in self.data

    def __iter__(self) -> Any:
        self._check_if_crashed()
        for k in self.data:
            yield self.data[k]


@serializable(recursive_serde=True)
class MockObjectType(SyftObject):
    __canonical_name__ = "mock_type"


@serializable(recursive_serde=True)
class MockStore(DocumentStore):
    pass


@serializable(recursive_serde=True)
class MockSyftObject(SyftObject):
    __canonical_name__ = UID()
    data: Any


@serializable(recursive_serde=True)
class MockStoreConfig(StoreConfig):
    store_type: Type[DocumentStore] = MockStore
    db_name: str = "testing"
    backing_store: Type[KeyValueBackingStore] = MockKeyValueBackingStore
    is_crashed: bool = False
