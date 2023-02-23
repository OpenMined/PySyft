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
from .policy import UserPolicy, SubmitUserPolicy

@serializable(recursive_serde=True)
class PolicyService(AbstractService):
    store: DocumentStore
    # policy_code_stash: PolicyStash
    policy_stash: UserPolicyStash
    
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        # self.policy_stash = PolicyStash(store=store)
        self.policy_code_stash = UserPolicyStash(store=store)
        
    # @service_method(path="policy.get_all", name="get_all")
    # def get_all_policies(
    #     self, context: AuthedServiceContext
    # ) -> Union[List[Policy], SyftError]:
    #     result = self.policy_stash.get_all()
    #     if result.is_ok():
    #         return result.ok()
    #     return SyftError(message=result.err())

    # @service_method(path="policy.create", name="create")
    # def create_policy(
    #     self, context: AuthedServiceContext, create_policy: CreatePolicy
    # ) -> Union[SyftSuccess, SyftError]:
    #     policy = create_policy.to(Policy, context=context)
    #     # TODO: check user_code uid
    #     policy_code = self.policy_code_stash.get_by_uid(policy["policy_code_uid"])
    #     if policy_code.is_err():
    #         return SyftError(message=str(result.err()))
    #     policy_code = policy_code.value
    #     exec(policy_code.raw_code)
    #     policy_class = eval(policy_code.class_name)
    #     policy_object = policy_class(**create_policy["policy_init_args"])
    #     policy.serde = _serialize(policy_object, to_bytes=True)
    #     result = self.policy_stash.set(policy)
    #     if result.is_err():
    #         return SyftError(message=str(result.err()))
    #     return SyftSuccess(message="Policy Created!")

    # @service_method(path="policy.get_by_id", name="get_by_id")
    # def get_policy_by_uid(
    #     self, context: AuthedServiceContext, uid: UID
    # ) -> Union[SyftSuccess, SyftError]:
    #     result = self.policy_stash.get_by_uid(uid=uid)
    #     if result.is_ok():
    #         return result.ok()
    #     return SyftError(message=result.err())
    
    # @service_method(path="policy.get_all_for_user", name="get_all_for_user")
    # def get_all_policies_for_user(
    #     self, context: AuthedServiceContext
    # ) -> Union[SyftSuccess, SyftError]:
    #     result = self.policy_stash.get_all_by_user_verify_key(user_verify_key=context.credentials)
    #     if result.is_ok():
    #         return result.ok()
    #     return SyftError(message=result.err())
    
    # @service_method(path="policy.get_public_state", name="get_public_state")
    # def get_public_state(
    #     self, context: AuthedServiceContext, uid: UID
    # ) -> Union[SyftSuccess, SyftError]:
    #     policy = self.policy_stash.get_by_uid(uid=uid)
    #     if policy.is_ok():
    #         policy = policy.ok()
    #         policy_code = self.policy_code_stash.get_by_uid(policy.policy_code_uid).value
    #         exec(policy_code.raw_code)
            
    #         policy_class_name = eval(policy_code.class_name)
        
    #         policy_object = _deserialize(policy.serde, from_bytes=True, class_type=policy_class_name)
    #         public_state = policy_object.public_state()
    #         return SyftSuccess(message=str(public_state))
    #     return SyftError(message=policy.err())
    
    @service_method(path="policy.run_code", name="run_code")
    def run_code(
        self, context: AuthedServiceContext, uid: UID, inputs: Dict[str, Any]
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        policy = self.policy_stash.get_by_uid(uid=uid)
        if policy.is_ok():
            policy = policy.ok()
            policy_code = self.policy_code_stash.get_by_uid(policy.policy_code_uid).value
            # print(policy_code.raw_code)
            exec(policy_code.raw_code)
            code_uid = policy.user_code_uid
            code_item = context.node.get_service(UserCodeService).get_by_uid(context, code_uid)
            exec_result = execute_byte_code(code_item, inputs)
            
            # print(exec_result.result)
            policy_class_name = eval(policy_code.class_name)
            # p = policy_class_name()
            # s = _serialize(p, to_bytes=True)
        
            policy_object = _deserialize(policy.serde, from_bytes=True, class_type=policy_class_name)
            # print(policy_object.public_state())

            policy_results = policy_object.apply_output(exec_result.result)
            # print(policy_object.public_state())
            serde = _serialize(policy_object, to_bytes=True)
            policy.serde = serde
            policy = self.policy_stash.update(policy=policy)
            return SyftSuccess(message=str(policy_results))
        return SyftError(message=policy.err())

    @service_method(path="policy_code.get_all", name="get_all")
    def get_all_policy_code(
        self, context: AuthedServiceContext
    ) -> Union[List[UserPolicy], SyftError]:
        result = self.policy_code_stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="policy_code.add", name="add")
    def add_policy_code(
        self, context: AuthedServiceContext, policy_code: SubmitUserPolicy
    ) -> Union[SyftSuccess, SyftError]:
        result = self.policy_code_stash.set(policy_code.to(UserPolicy, context=context))
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Policy Code Submitted")

    @service_method(path="policy_code.get_by_uid", name="get_by_uid")
    def get_policy_code_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.policy_code_stash.get_by_uid(uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())
    
    @service_method(path="policy_code.get_all_for_user", name="get_all_for_user")
    def get_all_policy_code_for_user(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        result = self.policy_code_stash.get_all_by_user_verify_key(user_verify_key=context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())