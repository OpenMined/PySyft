# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .api import TwinAPIEndpoint

MISSING_PATH_STRING = "Endpoint path: {path} does not exist."


@serializable(canonical_name="TwinAPIEndpointSQLStash", version=1)
class TwinAPIEndpointStash(ObjectStash[TwinAPIEndpoint]):
    @as_result(StashException, NotFoundException)
    def get_by_path(self, credentials: SyftVerifyKey, path: str) -> TwinAPIEndpoint:
        # TODO standardize by returning None if endpoint doesnt exist.
        res = self.get_one(
            credentials=credentials,
            filters={"path": path},
        )

        if res.is_err():
            raise NotFoundException(
                public_message=MISSING_PATH_STRING.format(path=path)
            )
        return res.unwrap()

    @as_result(StashException)
    def path_exists(self, credentials: SyftVerifyKey, path: str) -> bool:
        try:
            self.get_by_path(credentials=credentials, path=path).unwrap()
            return True
        except NotFoundException:
            return False
