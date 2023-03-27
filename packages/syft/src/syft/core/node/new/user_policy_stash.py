# stdlib
from typing import List

# third party
from result import Result

# relative
from .credentials import SyftVerifyKey
from .document_store import BaseUIDStoreStash
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .policy import PolicyUserVerifyKeyPartitionKey
from .new_policy import UserPolicy
from .serializable import serializable


@serializable(recursive_serde=True)
class UserPolicyStash(BaseUIDStoreStash):
    object_type = UserPolicy
    settings: PartitionSettings = PartitionSettings(
        name=UserPolicy.__canonical_name__, object_type=UserPolicy
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_all_by_user_verify_key(
        self, user_verify_key: SyftVerifyKey
    ) -> Result[List[UserPolicy], str]:
        qks = QueryKeys(qks=[PolicyUserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(qks=qks)
