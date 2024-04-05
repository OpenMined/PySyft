# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from .api import TwinAPIEndpoint

MISSING_PATH_STRING = "Endpoint path: {path} does not exist."


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

        endpoints = []
        if endpoint_results.is_ok():
            endpoints = endpoint_results.ok()

        for endpoint in endpoints:
            if endpoint.path == path:
                return Ok(endpoint)

        return Err(MISSING_PATH_STRING.format(path=path))

    def path_exists(self, credentials: SyftVerifyKey, path: str) -> Result[bool, str]:
        result = self.get_by_path(credentials=credentials, path=path)

        if result.is_err() and result.err() == MISSING_PATH_STRING.format(path=path):
            return Ok(False)
        if result.is_ok():
            return Ok(True)

        return Err(result.err())

    def upsert(
        self,
        credentials: SyftVerifyKey,
        endpoint: TwinAPIEndpoint,
        has_permission: bool = False,
    ) -> Result[TwinAPIEndpoint, str]:
        """Upsert an endpoint."""
        result = self.path_exists(credentials=credentials, path=endpoint.path)

        if result.is_err():
            return result

        if result.ok():
            super().delete_by_uid(credentials=credentials, uid=endpoint.id)

        result = super().set(
            credentials=credentials, obj=endpoint, ignore_duplicates=False
        )
        return result
