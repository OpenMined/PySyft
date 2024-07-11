# stdlib

# third party
from result import as_result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionSettings
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from .api import TwinAPIEndpoint

MISSING_PATH_STRING = "Endpoint path: {path} does not exist."


@serializable()
class TwinAPIEndpointStash(NewBaseUIDStoreStash):
    object_type = TwinAPIEndpoint
    settings: PartitionSettings = PartitionSettings(
        name=TwinAPIEndpoint.__canonical_name__, object_type=TwinAPIEndpoint
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(StashException, NotFoundException)
    def get_by_path(self, credentials: SyftVerifyKey, path: str) -> TwinAPIEndpoint:
        endpoints = self.get_all(credentials=credentials).unwrap()
        for endpoint in endpoints:
            if endpoint.path == path:
                return endpoint

        raise NotFoundException(public_message=MISSING_PATH_STRING.format(path=path))

    @as_result(StashException)
    def path_exists(self, credentials: SyftVerifyKey, path: str) -> bool:
        try:
            result = self.get_by_path(credentials=credentials, path=path).unwrap()
            return False
        except NotFoundException:
            return True

    @as_result(StashException)
    def upsert(
        self,
        credentials: SyftVerifyKey,
        endpoint: TwinAPIEndpoint,
        has_permission: bool = False,
    ) -> TwinAPIEndpoint:
        """Upsert an endpoint."""
        path_exists = self.path_exists(
            credentials=credentials, path=endpoint.path
        ).unwrap()

        if path_exists:
            super().delete_by_uid(credentials=credentials, uid=endpoint.id).unwrap()

        return (
            super()
            .set(credentials=credentials, obj=endpoint, ignore_duplicates=False)
            .unwrap()
        )
