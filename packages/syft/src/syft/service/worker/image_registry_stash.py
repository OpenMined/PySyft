# stdlib

# third party
from result import Ok, Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import (
    BaseUIDStoreStash,
    DocumentStore,
    PartitionKey,
    PartitionSettings,
    QueryKeys,
)
from ..response import SyftSuccess
from .image_registry import SyftImageRegistry

__all__ = ["SyftImageRegistryStash"]


URLPartitionKey = PartitionKey(key="url", type_=str)


@serializable(canonical_name="SyftImageRegistryStash", version=1)
class SyftImageRegistryStash(BaseUIDStoreStash):
    object_type = SyftImageRegistry
    settings: PartitionSettings = PartitionSettings(
        name=SyftImageRegistry.__canonical_name__,
        object_type=SyftImageRegistry,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_url(
        self,
        credentials: SyftVerifyKey,
        url: str,
    ) -> Result[SyftImageRegistry | None, str]:
        qks = QueryKeys(qks=[URLPartitionKey.with_obj(url)])
        return self.query_one(credentials=credentials, qks=qks)

    def delete_by_url(
        self, credentials: SyftVerifyKey, url: str,
    ) -> Result[SyftSuccess, str]:
        qk = URLPartitionKey.with_obj(url)
        result = super().delete(credentials=credentials, qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"URL: {url} deleted"))
        return result
