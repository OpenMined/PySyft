from typing import List
from typing import Optional

from result import Result

from ...common.serde.serializable import serializable
from .credentials import SyftVerifyKey

from .document_store import BaseUIDStoreStash
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .document_store import DocumentStore
from .policy import PolicyCode
from .policy import UserVerifyKeyPartitionKey


@serializable(recursive_serde=True)
class PolicyCodeStash(BaseUIDStoreStash):
    object_type = PolicyCode
    settings: PartitionSettings = PartitionSettings(
        name=PolicyCode.__canonical_name__, object_type=PolicyCode
    )
    
    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
        

    def get_all_by_user_verify_key(
        self, user_verify_key: SyftVerifyKey
    ) -> Result[List[PolicyCode], str]:
        qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(qks=qks)