from typing import Union
from typing import List
from typing import Dict
from typing import Any

from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize
from ...common.serde.deserialize import _deserialize
from ...common.uid import UID
from .context import AuthedServiceContext
# from .policy import Policy, CreatePolicy
from .service import AbstractService
from .service import service_method
from .response import SyftError
from .response import SyftSuccess
from .document_store import DocumentStore
# from .policy_stash import PolicyStash
from .user_code import execute_byte_code
from .user_code_service import UserCodeService
from .user_policy_stash import UserPolicyStash
from .policy import UserPolicy, SubmitUserPolicy, ExactMatch, SingleExecutionExactOutput

# TODO: replace policy_code with user_policy
@serializable(recursive_serde=True)
class PolicyService(AbstractService):
    store: DocumentStore
    # policy_code_stash: PolicyStash
    policy_stash: UserPolicyStash
    
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        # self.policy_stash = PolicyStash(store=store)
        self.policy_stash = UserPolicyStash(store=store)

    @service_method(path="policy.get_all", name="get_all")
    def get_all_user_policy(
        self, context: AuthedServiceContext
    ) -> Union[List[UserPolicy], SyftError]:
        result = self.policy_stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="policy.add", name="add")
    def add_user_policy(
        self, context: AuthedServiceContext, policy_code: Union[SubmitUserPolicy, UserPolicy]
    ) -> Union[SyftSuccess, SyftError]:
        if isinstance(policy_code, SubmitUserPolicy):
            policy_code = policy_code.to(UserPolicy, context=context)
        result = self.policy_stash.set(policy_code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Policy Code Submitted")
    

    @service_method(path="policy.get_by_uid", name="get_by_uid")
    def get_policy_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.policy_stash.get_by_uid(uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    
# move them in the database
ALLOWED_POLICIES = [ExactMatch, SingleExecutionExactOutput]