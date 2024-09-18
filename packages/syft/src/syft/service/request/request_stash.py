# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from .request import Request


@serializable(canonical_name="RequestStashSQL", version=1)
class RequestStash(ObjectStash[Request]):
    @as_result(SyftException)
    def get_all_for_verify_key(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
    ) -> list[Request]:
        return self.get_all(
            credentials=credentials,
            filters={"requesting_user_verify_key": verify_key},
        ).unwrap()

    @as_result(SyftException)
    def get_by_usercode_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> list[Request]:
        return self.get_all(
            credentials=credentials,
            filters={"code_id": user_code_id},
        ).unwrap()
