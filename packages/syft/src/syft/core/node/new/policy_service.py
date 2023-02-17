from typing import Union
from typing import List
from typing import Dict
from typing import Any

from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize
from ...common.serde.deserialize import _deserialize
from ...common.uid import UID
from .context import AuthedServiceContext
from .policy import Policy, CreatePolicy
from .service import AbstractService
from .service import service_method
from .response import SyftError
from .response import SyftSuccess
from .document_store import DocumentStore
from .policy_stash import PolicyStash
from .user_code import execute_byte_code

@serializable(recursive_serde=True)
class PolicyService(AbstractService):
    store: DocumentStore
    stash: PolicyStash
    
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = PolicyStash(store=store)
        
    @service_method(path="policy.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[Policy], SyftError]:
        result = self.stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="policy.create", name="create")
    def create(
        self, context: AuthedServiceContext, create_policy: CreatePolicy
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.set(create_policy.to(Policy, context=context))
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Policy Created!")

    @service_method(path="policy.get_by_id", name="get_by_id")
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())
    
    @service_method(path="policy.get_all_for_user", name="get_all_for_user")
    def get_all_for_user(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_all_by_user_verify_key(user_verify_key=context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())
        
    def run_code(
        self, context: AuthedServiceContext, uid: UID, inputs: Dict[str, Any]
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        policy = self.stash.get_by_uid(uid=uid)
        if policy.is_ok():
            code_item = policy.ok()
            exec_result = execute_byte_code(code_item, inputs)
            policy_object = _deserialize(policy["serde"], from_bytes=True)
            policy_results = policy_object.apply_output(exec_result.result)
            serde = _serialize(policy_object, from_bytes=True)
            policy["serde"] = serde
            self.stash.udpate(policy)
            return policy_results
        return SyftError(message=policy.err())

