# stdlib

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .policy import UserPolicy


@serializable(canonical_name="UserPolicySQLStash", version=1)
class UserPolicyStash(ObjectStash[UserPolicy]):
    @as_result(StashException, NotFoundException)
    def get_all_by_user_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> list[UserPolicy]:
        return self.get_all(
            credentials=credentials,
            filters={"user_verify_key": user_verify_key},
        ).unwrap()
