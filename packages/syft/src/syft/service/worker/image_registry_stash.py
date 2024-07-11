# stdlib

# third party
from result import Ok
from result import Result
from syft.store.document_store_errors import NotFoundException, StashException
from syft.types.result import as_result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash, NewBaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ..response import SyftSuccess
from .image_registry import SyftImageRegistry

__all__ = ["SyftImageRegistryStash"]


URLPartitionKey = PartitionKey(key="url", type_=str)


@serializable()
class SyftImageRegistryStash(NewBaseUIDStoreStash):
    object_type = SyftImageRegistry
    settings: PartitionSettings = PartitionSettings(
        name=SyftImageRegistry.__canonical_name__,
        object_type=SyftImageRegistry,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
    
    @as_result(StashException, NotFoundException)
    def get_by_url(
        self,
        credentials: SyftVerifyKey,
        url: str,
    ) -> Result[SyftImageRegistry | None, str]:
        qks = QueryKeys(qks=[URLPartitionKey.with_obj(url)])
        try:
            return self.query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundException as exc:
            raise NotFoundException.from_exception(exc, public_message=f"Image Registry with url {url} not found")

    @as_result(StashException)
    def delete_by_url(
        self, credentials: SyftVerifyKey, url: str
    ) -> Result[SyftSuccess, str]:
        qk = URLPartitionKey.with_obj(url)
        return super().delete(credentials=credentials, qk=qk).unwrap()