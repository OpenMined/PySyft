# stdlib

# third party

# stdlib

# stdlib
from typing import Literal

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from .image_registry import SyftImageRegistry

__all__ = ["SyftImageRegistryStash"]


URLPartitionKey = PartitionKey(key="url", type_=str)


@serializable(canonical_name="SyftImageRegistryStash", version=1)
class SyftImageRegistryStash(NewBaseUIDStoreStash):
    object_type = SyftImageRegistry
    settings: PartitionSettings = PartitionSettings(
        name=SyftImageRegistry.__canonical_name__,
        object_type=SyftImageRegistry,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(SyftException, StashException, NotFoundException)
    def get_by_url(
        self,
        credentials: SyftVerifyKey,
        url: str,
    ) -> SyftImageRegistry | None:
        qks = QueryKeys(qks=[URLPartitionKey.with_obj(url)])
        return self.query_one(credentials=credentials, qks=qks).unwrap(
            public_message=f"Image Registry with url {url} not found"
        )

    @as_result(SyftException, StashException)
    def delete_by_url(self, credentials: SyftVerifyKey, url: str) -> Literal[True]:
        qk = URLPartitionKey.with_obj(url)
        return super().delete(credentials=credentials, qk=qk).unwrap()
