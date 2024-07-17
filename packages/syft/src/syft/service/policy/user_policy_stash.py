# stdlib

# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from .policy import PolicyUserVerifyKeyPartitionKey
from .policy import UserPolicy


@serializable(canonical_name="UserPolicyStash", version=1)
class UserPolicyStash(BaseUIDStoreStash):
    object_type = UserPolicy
    settings: PartitionSettings = PartitionSettings(
        name=UserPolicy.__canonical_name__, object_type=UserPolicy
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_all_by_user_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> Result[list[UserPolicy], str]:
        qks = QueryKeys(qks=[PolicyUserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(credentials=credentials, qks=qks)
