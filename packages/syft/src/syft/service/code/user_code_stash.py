# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .user_code import UserCode


@serializable(canonical_name="UserCodeSQLStash", version=1)
class UserCodeStash(ObjectStash[UserCode]):
    @as_result(StashException, NotFoundException)
    def get_by_code_hash(self, credentials: SyftVerifyKey, code_hash: str) -> UserCode:
        return self.get_one(
            credentials=credentials,
            filters={"code_hash": code_hash},
        ).unwrap()

    @as_result(StashException)
    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> list[UserCode]:
        return self.get_all(
            credentials=credentials,
            filters={"service_func_name": service_func_name},
        ).unwrap()
