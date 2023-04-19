# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..context import UnauthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from .metadata_stash import MetadataStash
from .node_metadata import NodeMetadata
from .node_metadata import NodeMetadataUpdate


@serializable()
class MetadataService(AbstractService):
    store: DocumentStore
    stash: MetadataStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = MetadataStash(store=store)

    @service_method(path="metadata.get", name="get")
    def get(self, context: UnauthedServiceContext) -> Result[Ok, Err]:
        """Get Metadata"""
        result = self.stash.get_all(context.node.signing_key.verify_key)
        if result.is_ok():
            metadata = result.ok()
            # check if the metadata list is empty
            if len(metadata) == 0:
                return SyftError(message="No metadata found")
            result = metadata[0]
            return Ok(result)
        else:
            return SyftError(message=result.err())

    @service_method(path="metadata.set", name="set")
    def set(
        self, context: AuthedServiceContext, metadata: NodeMetadata
    ) -> Result[Ok, Err]:
        """Set a new the Node Metadata"""
        result = self.stash.set(context.credentials, metadata)
        if result.is_ok():
            return result
        else:
            return SyftError(message=result.err())

    @service_method(path="metadata.update", name="update")
    def update(
        self, context: AuthedServiceContext, metadata: NodeMetadataUpdate
    ) -> Result[Ok, Err]:
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            current_metadata = result.ok()
            if len(current_metadata) > 0:
                new_metadata = current_metadata[0].copy(
                    update=metadata.dict(exclude_unset=True)
                )
                update_result = self.stash.update(context.credentials, new_metadata)
                if update_result.is_ok():
                    return result
                else:
                    return SyftError(message=update_result.err())
            else:
                return SyftError(message="No metadata found")
        else:
            return SyftError(message=result.err())
