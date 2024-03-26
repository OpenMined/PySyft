# stdlib

# third party
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from .api import TwinAPIEndpoint


@serializable()
class TwinAPIEndpointStash(BaseUIDStoreStash):
    object_type = TwinAPIEndpoint
    settings: PartitionSettings = PartitionSettings(
        name=TwinAPIEndpoint.__canonical_name__, object_type=TwinAPIEndpoint
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_path(
        self, credentials: SyftVerifyKey, path: str
    ) -> Result[TwinAPIEndpoint, str]:
        endpoint_results = self.get_all(credentials=credentials)

        if endpoint_results.is_err():
            return endpoint_results

        endpoint_by_path = next(
            (endpoint for endpoint in endpoint_results.ok() if endpoint.path == path),
            None,
        )

        return Ok(endpoint_by_path)

    def upsert(
        self,
        credentials: SyftVerifyKey,
        endpoint: TwinAPIEndpoint,
        has_permission: bool = False,
    ) -> Result[TwinAPIEndpoint, str]:
        """Upsert an endpoint."""

        result = self.get_by_path(credentials=credentials, path=endpoint.path)

        if result.is_err():
            return result

        if result.ok():
            super().delete_by_uid(credentials=credentials, uid=result.ok().id)

        result = super().set(
            credentials=credentials, obj=endpoint, ignore_duplicates=False
        )
        return result
