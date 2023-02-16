from typing import Union
from typing import List

from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .policy_code import PolicyCode, SubmitPolicyCode
from .service import AbstractService
from .service import service_method
from .response import SyftError
from .response import SyftSuccess
from .document_store import DocumentStore
from .policy_code_stash import PolicyCodeStash

@serializable(recursive_serde=True)
class PolicyCodeService(AbstractService):
    store: DocumentStore
    stash: PolicyCodeStash
    
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = PolicyCodeStash(store=store)
        
    @service_method(path="policy.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[PolicyCode], SyftError]:
        result = self.stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="policy.add", name="add")
    def add(
        self, context: AuthedServiceContext, policy_code: SubmitPolicyCode
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.set(policy_code.to(PolicyCode, context=context))
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Policy Code Submitted")

    @service_method(path="policy_code.get_by_id", name="get_by_id")
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())
    
    @service_method(path="policy_code.get_all_for_user", name="get_all_for_user")
    def get_all_for_user(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_all_by_user_verify_key(user_verify_key=context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())
        