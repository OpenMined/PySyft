# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# relative
from ....telemetry import instrument
from ...common.serde import _serialize
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .document_store import DocumentStore
from .linked_obj import LinkedObject
from .policy import InputPolicy
from .policy import OutputPolicy
from .policy import SubmitUserPolicy
from .policy import UserPolicy
from .policy import get_policy_object
from .policy import init_policy
from .policy import update_policy_state
from .response import SyftError
from .response import SyftNotReady
from .response import SyftSuccess
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
from .service import service_method
from .user_code import SubmitUserCode
from .user_code import UserCode
from .user_code import UserCodeStatus
from .user_code_stash import UserCodeStash


@instrument
@serializable(recursive_serde=True)
class UserCodeService(AbstractService):
    store: DocumentStore
    stash: UserCodeStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = UserCodeStash(store=store)

    @service_method(path="code.submit", name="submit")
    def submit(
        self, context: AuthedServiceContext, code: SubmitUserCode
    ) -> Union[UserCode, SyftError]:
        """Add User Code"""
        code_item = code.to(UserCode, context=context)
        # relative
        from .policy_service import PolicyService

        policy_service = context.node.get_service(PolicyService)

        if isinstance(code.input_policy, SubmitUserPolicy):
            submit_input_policy = code.input_policy
            code_item.input_policy = submit_input_policy.to(UserPolicy, context=context)
        elif isinstance(code.input_policy, UID):
            input_policy = policy_service.get_policy_by_uid(context, code.input_policy)
            if input_policy.is_ok():
                code_item.input_policy = input_policy.ok()
            else:
                return input_policy

        if isinstance(code.output_policy, SubmitUserPolicy):
            submit_output_policy = code.output_policy
            code_item.output_policy = submit_output_policy.to(
                UserPolicy, context=context
            )
        elif isinstance(code.output_policy, UID):
            output_policy = policy_service.policy_stash.get_by_uid(code.output_policy)
            if output_policy.is_ok():
                code_item.output_policy = output_policy.ok()
            else:
                return output_policy

        result = self.stash.set(code_item)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Submitted")

    @service_method(path="code.request_code_execution", name="request_code_execution")
    def request_code_execution(
        self, context: AuthedServiceContext, code: SubmitUserCode
    ) -> Union[SyftSuccess, SyftError]:
        """Request Code execution on user code"""
        # relative
        from .request import EnumMutation
        from .request import SubmitRequest
        from .request_service import RequestService

        user_code = code.to(UserCode, context=context)

        result = self.stash.set(user_code)
        if result.is_err():
            return SyftError(message=str(result.err()))

        linked_obj = LinkedObject.from_obj(user_code, node_uid=context.node.id)
        CODE_EXECUTE = EnumMutation.from_obj(
            linked_obj=linked_obj, attr_name="status", value=UserCodeStatus.EXECUTE
        )
        request = SubmitRequest(changes=[CODE_EXECUTE])
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request)

        # The Request service already returns either a SyftSuccess or SyftError
        return result

    @service_method(path="code.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[UserCode], SyftError]:
        """Get a Dataset"""
        result = self.stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code.get_by_id", name="get_by_id")
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get a User Code Item"""
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code.get_all_for_user", name="get_all_for_user")
    def get_all_for_user(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        """Get All User Code Items for User's VerifyKey"""
        # TODO: replace with incoming user context and key
        result = self.stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def update_code_state(
        self, context: AuthedServiceContext, code_item: UserCode
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.update(code_item)
        if result.is_ok():
            return SyftSuccess(message="Code State Updated")
        return SyftError(message="Unable to Update Code State")

    @service_method(path="code.review", name="review")
    def review(self, context: AuthedServiceContext, uid: UID, approved: bool):
        # TODO: Check for permissions
        # TODO: for some reason the output is a SyftError, even tho everything works
        # relative
        from .policy_service import PolicyService

        policy_service = context.node.get_service(PolicyService)
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            code_item = result.ok()
            if approved:
                code_item.status = UserCodeStatus.EXECUTE
                if isinstance(code_item.input_policy, UserPolicy):
                    policy_service.add_user_policy(context, code_item.input_policy)
                    policy_object = init_policy(
                        code_item.input_policy, code_item.input_policy_init_args
                    )
                    code_item.input_policy_state = _serialize(
                        policy_object, to_bytes=True
                    )

                if isinstance(code_item.output_policy, UserPolicy):
                    policy_service.add_user_policy(context, code_item.output_policy)
                    policy_object = init_policy(
                        code_item.output_policy, code_item.output_policy_init_args
                    )
                    code_item.output_policy_state = _serialize(
                        policy_object, to_bytes=True
                    )
            else:
                code_item.status = UserCodeStatus.DENIED

            self.update_code_state(context=context, code_item=code_item)

            return SyftSuccess("Review submitted")

        return SyftError(message=result.err())

    @service_method(path="code.call", name="call")
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        filtered_kwargs = filter_kwargs(kwargs)
        try:
            result = self.stash.get_by_uid(uid=uid)
            if result.is_ok():
                code_item = result.ok()
                if code_item.status == UserCodeStatus.EXECUTE:
                    is_valid = True  # = code_item.output_policy_state.valid
                    if not is_valid:
                        return is_valid
                    else:
                        action_service = context.node.get_service("actionservice")
                        if isinstance(code_item.input_policy, InputPolicy):
                            # TODO: fix bug with dev InputPolicy
                            # filtered_kwargs = code_item.input_policy.filter_kwargs(filtered_kwargs)
                            filtered_kwargs = filtered_kwargs
                        else:
                            policy_object = get_policy_object(
                                code_item.input_policy, code_item.input_policy_state
                            )
                            filtered_kwargs = policy_object.filter_kwargs(
                                filtered_kwargs
                            )
                            code_item.input_policy_state = update_policy_state(
                                policy_object
                            )

                        result = action_service._user_code_execute(
                            context, code_item, filtered_kwargs
                        )
                        if isinstance(result, str):
                            return SyftError(message=result)
                        if result.is_ok():
                            final_results = result.ok()
                            if isinstance(code_item.output_policy, OutputPolicy):
                                code_item.output_policy_state.update_state()
                            else:
                                policy_object = get_policy_object(
                                    code_item.output_policy,
                                    code_item.output_policy_state,
                                )

                                final_results = policy_object.apply_output(
                                    final_results
                                )
                                code_item.output_policy_state = update_policy_state(
                                    policy_object
                                )

                            state_result = self.update_code_state(
                                context=context, code_item=code_item
                            )
                            if state_result:
                                return final_results
                            else:
                                return state_result
                elif code_item.status == UserCodeStatus.SUBMITTED:
                    return SyftNotReady(
                        message=f"{type(code_item)} Your code is waiting for approval: {code_item.status}"
                    )
                else:
                    return SyftError(
                        message=f"{type(code_item)} Your code cannot be run: {code_item.status}"
                    )
            return SyftError(message=result.err())
        except Exception as e:
            return SyftError(message=f"Failed to run. {e}")


def filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # relative
    from .action_object import ActionObject
    from .dataset import Asset
    from .twin_object import TwinObject

    filtered_kwargs = {}
    for k, v in kwargs.items():
        value = v
        if isinstance(v, ActionObject):
            value = v.id
        if isinstance(v, TwinObject):
            value = v.id
        if isinstance(v, Asset):
            value = v.action_id
        filtered_kwargs[k] = value
    return filtered_kwargs


TYPE_TO_SERVICE[UserCode] = UserCodeService
SERVICE_TO_TYPES[UserCodeService].update({UserCode})
