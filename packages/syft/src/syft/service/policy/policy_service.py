# stdlib

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .policy import SubmitUserPolicy
from .policy import UserPolicy
from .user_policy_stash import UserPolicyStash


@serializable(canonical_name="PolicyService", version=1)
class PolicyService(AbstractService):
    stash: UserPolicyStash

    def __init__(self, store: DBManager) -> None:
        self.stash = UserPolicyStash(store=store)

    @service_method(path="policy.get_all", name="get_all")
    def get_all_user_policy(self, context: AuthedServiceContext) -> list[UserPolicy]:
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(path="policy.add", name="add", unwrap_on_success=False)
    def add_user_policy(
        self,
        context: AuthedServiceContext,
        policy_code: SubmitUserPolicy | UserPolicy,
    ) -> SyftSuccess:
        if isinstance(policy_code, SubmitUserPolicy):
            policy_code = policy_code.to(UserPolicy, context=context)
        result = self.stash.set(context.credentials, policy_code).unwrap()
        return SyftSuccess(message="Policy Code Submitted", value=result)

    @service_method(path="policy.get_by_uid", name="get_by_uid")
    def get_policy_by_uid(self, context: AuthedServiceContext, uid: UID) -> UserPolicy:
        return self.stash.get_by_uid(context.credentials, uid=uid).unwrap()


TYPE_TO_SERVICE[UserPolicy] = UserPolicy
