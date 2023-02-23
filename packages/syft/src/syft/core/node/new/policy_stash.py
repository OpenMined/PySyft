# from typing import List

# from result import Result

# from ...common.serde.serializable import serializable
# from .credentials import SyftVerifyKey
# from .document_store import BaseUIDStoreStash
# from .document_store import PartitionSettings
# from .document_store import QueryKeys
# from .document_store import DocumentStore
# from .document_store import UIDPartitionKey

# from .policy import Policy
# from .policy import UserVerifyKeyPartitionKey


# @serializable(recursive_serde=True)
# class PolicyStash(BaseUIDStoreStash):
#     object_type = Policy
#     settings: PartitionSettings = PartitionSettings(
#         name=Policy.__canonical_name__, object_type=Policy
#     )
    
#     def __init__(self, store: DocumentStore) -> None:
#         super().__init__(store=store)

#     def get_all_by_user_verify_key(
#         self, user_verify_key: SyftVerifyKey
#     ) -> Result[List[Policy], str]:
#         qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(user_verify_key)])
#         return self.query_one(qks=qks)
    
#     def update(self, policy: Policy) -> Result[Policy, str]:
#         return self.check_type(policy, self.object_type).and_then(super().update)
