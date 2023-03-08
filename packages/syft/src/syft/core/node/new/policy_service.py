# stdlib
from typing import List
from typing import Union

# relative
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .document_store import DocumentStore
from .user_code import ExactMatch
from .user_code import SingleExecutionExactOutput
from .policy import SubmitUserPolicy
from .policy import UserPolicy
from .response import SyftError
from .response import SyftSuccess

# from .policy import Policy, CreatePolicy
from .service import AbstractService
from .service import service_method

# from .policy_stash import PolicyStash
from .user_policy_stash import UserPolicyStash


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
        self,
        context: AuthedServiceContext,
        policy_code: Union[SubmitUserPolicy, UserPolicy],
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
