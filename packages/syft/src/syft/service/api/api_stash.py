# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from .api import TwinAPIEndpoint

MISSING_PATH_STRING = "Endpoint path: {path} does not exist."


@serializable(canonical_name="TwinAPIEndpointSQLStash", version=1)
class TwinAPIEndpointStash(ObjectStash[TwinAPIEndpoint]):
    object_type = TwinAPIEndpoint
    settings: PartitionSettings = PartitionSettings(
        name=TwinAPIEndpoint.__canonical_name__, object_type=TwinAPIEndpoint
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_path(
        self, credentials: SyftVerifyKey, path: str
    ) -> Result[TwinAPIEndpoint, str]:
        # TODO standardize by returning None if endpoint doesnt exist.
        res_or_err = self.get_one_by_field(
            credentials=credentials,
            field_name="path",
            field_value=path,
        )

        if res_or_err.is_err():
            return res_or_err

        res = res_or_err.ok()
        if res is None:
            return Err(MISSING_PATH_STRING.format(path=path))
        return Ok(res)

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
        # TODO has_permission is not used.
        result = self.path_exists(credentials=credentials, path=endpoint.path)

        if result.is_err():
            return result

        if result.ok():
            super().delete_by_uid(credentials=credentials, uid=endpoint.id)

        result = super().set(
            credentials=credentials, obj=endpoint, ignore_duplicates=False
        )
        return result
