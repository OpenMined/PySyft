# stdlib
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from .object_metadata import SyftObjectMetadata
from .object_metadata import SyftObjectMetadataStash


@serializable()
class ObjectSearchService(AbstractService):
    store: DocumentStore
    stash: SyftObjectMetadata

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash: SyftObjectMetadataStash = SyftObjectMetadataStash(store=store)

    @service_method(path="object_metadata", name="search")
    def search(
        self, context: AuthedServiceContext, canonical_name: str
    ) -> Union[SyftObjectMetadata, SyftError]:
        """Search for the metadata for an object."""

        result = self.stash.get_by_name(
            canonical_name=canonical_name, credentials=context.credentials
        )

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        result = result.ok()

        if result is None:
            return SyftError(
                message=f"No metadata exists for canonical name: {canonical_name}"
            )

        return result
