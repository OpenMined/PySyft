# stdlib

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .policy import PolicyUserVerifyKeyPartitionKey
from .policy import UserPolicy


@serializable(canonical_name="UserPolicyStash", version=1)
class UserPolicyStash(NewBaseUIDStoreStash):
    object_type = UserPolicy
    settings: PartitionSettings = PartitionSettings(
        name=UserPolicy.__canonical_name__, object_type=UserPolicy
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(StashException, NotFoundException)
    def get_all_by_user_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> list[UserPolicy]:
        qks = QueryKeys(qks=[PolicyUserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()
