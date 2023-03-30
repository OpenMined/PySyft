# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
from result import OkErr

# relative
from ....telemetry import instrument
from .context import AuthedServiceContext
from .document_store import DocumentStore
from .linked_obj import LinkedObject
from .policy import OutputHistory
from .policy import OutputPolicy
from .policy import UserPolicy
from .policy import UserPolicyStatus
from .policy import get_policy_object
from .policy import init_policy
from .policy import load_policy_code
from .policy import update_policy_state
from .policy_service import PolicyService
from .request import EnumMutation
from .request import SubmitRequest
from .request import UserCodeStatusChange
from .request_service import RequestService
from .response import SyftError
from .response import SyftNotReady
from .response import SyftSuccess
from .serializable import serializable
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
from .service import service_method
from .twin_object import TwinObject
from .uid import UID
from .user_code import SubmitUserCode
from .user_code import UserCode
from .user_code import UserCodeStatus
from .user_code_stash import UserCodeStash
from .user_roles import GUEST_ROLE_LEVEL


@instrument
@serializable()
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
        user_code = code.to(UserCode, context=context)
        result = self.stash.set(user_code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Submitted")

    def _code_execution(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode,
    ):
        print("trying to conver object")
        user_code = code.to(UserCode, context=context)
        print("converted object", user_code)
        result = self.stash.set(user_code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        policy_service = context.node.get_service(PolicyService)

        linked_obj = LinkedObject.from_obj(user_code, node_uid=context.node.id)

        CODE_EXECUTE = UserCodeStatusChange(
            value=UserCodeStatus.EXECUTE, linked_obj=linked_obj
        )
        changes = [CODE_EXECUTE]

        # if isinstance(user_code.input_policy, UserPolicy):
        #     policy_service.add_user_policy(context, user_code.input_policy)
        #     input_policy_linked_obj = LinkedObject.from_obj(
        #         user_code.input_policy,
        #         node_uid=context.node.id,
        #         service_type=PolicyService,
        #     )
        #     INPUT_POLICY_APPROVE = EnumMutation(
        #         linked_obj=input_policy_linked_obj,
        #         attr_name="status",
        #         enum_type=UserPolicyStatus,
        #         value=UserPolicyStatus.APPROVED,
        #     )
        #     changes.append(INPUT_POLICY_APPROVE)

        # if isinstance(user_code.output_policy, UserPolicy):
        #     policy_service.add_user_policy(context, user_code.output_policy)
        #     output_policy_linked_obj = LinkedObject.from_obj(
        #         user_code.output_policy,
        #         node_uid=context.node.id,
        #         service_type=PolicyService,
        #     )
        #     OUTPUT_POLICY_APPROVE = EnumMutation(
        #         linked_obj=output_policy_linked_obj,
        #         attr_name="status",
        #         enum_type=UserPolicyStatus,
        #         value=UserPolicyStatus.APPROVED,
        #     )
        #     changes.append(OUTPUT_POLICY_APPROVE)

        request = SubmitRequest(changes=changes)
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request)

        # The Request service already returns either a SyftSuccess or SyftError
        return result

    @service_method(
        path="code.request_code_execution",
        name="request_code_execution",
        roles=GUEST_ROLE_LEVEL,
    )
    def request_code_execution(
        self, context: AuthedServiceContext, code: SubmitUserCode
    ) -> Union[SyftSuccess, SyftError]:
        """Request Code execution on user code"""
        return self._code_execution(context=context, code=code)

    @service_method(path="code.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
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

    def load_user_code(self) -> None:
        result = self.stash.get_all()
        if result.is_ok():
            user_code_items = result.ok()
            print("user_code_items", user_code_items)
            for user_code in user_code_items:
                if user_code.status.approved:
                    print(
                        "!!!! user_code.input_policy_type",
                        user_code.input_policy_type,
                        type(user_code.input_policy_type),
                    )
                    print(
                        "!!!!! user_code.output_policy_type",
                        user_code.output_policy_type,
                        type(user_code.output_policy_type),
                    )
                    if isinstance(user_code.input_policy_type, UserPolicy):
                        load_policy_code(user_code.input_policy_type)
                    if isinstance(user_code.output_policy_type, UserPolicy):
                        load_policy_code(user_code.output_policy_type)
        print("finished loading user code")

    @service_method(path="code.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        try:
            filtered_kwargs = filter_kwargs(kwargs)
            result = self.stash.get_by_uid(uid=uid)
            if not result.is_ok():
                return SyftError(message=result.err())

            # Unroll variables
            code_item = result.ok()
            output_policy = code_item.output_policy
            if output_policy is None:
                raise Exception(f"Output policy not approved", code_item)
            op_state = code_item.output_policy_state
            op_init_args = code_item.output_policy_init_kwargs
            status = code_item.status

            # Check if we are allowed to execute the code
            if status.for_context(context) != UserCodeStatus.EXECUTE:
                if status.for_context(context) == UserCodeStatus.SUBMITTED:
                    return SyftNotReady(
                        message=f"{type(code_item)} Your code is waiting for approval: {status}"
                    )
                return SyftError(
                    message=f"{type(code_item)} Your code cannot be run: {status}"
                )

            # Check if the OutputPolicy is valid
            is_valid = output_policy.valid

            if not is_valid:
                if len(output_policy.output_history) > 0:
                    return get_outputs(
                        context=context,
                        output_history=output_policy.output_history[-1],
                    )
                return is_valid

            # Execute the code item
            action_service = context.node.get_service("actionservice")
            result = action_service._user_code_execute(
                context, code_item, filtered_kwargs
            )
            if isinstance(result, str):
                return SyftError(message=result)

            # Apply Output Policy to the results and update the OutputPolicyState
            final_results = result.ok()
            if isinstance(output_policy, OutputPolicy) and not isinstance(
                op_state, bytes
            ):
                op_state.apply_output(context=context, outputs=final_results)
            else:
                # For User Policies we need to get the object from the byte_code
                if hasattr(output_policy, "byte_code"):
                    if len(op_state) == 0:
                        policy_object = init_policy(output_policy, op_init_args)
                    else:
                        policy_object = get_policy_object(output_policy, op_state)
                else:
                    policy_object = code_item.output_policy

                policy_object.apply_output(context, final_results)
                if hasattr(code_item.output_policy, "byte_code"):
                    code_item.output_policy_state = update_policy_state(policy_object)

            state_result = self.update_code_state(context=context, code_item=code_item)
            if state_result:
                # TODO: there are probably circumstances where we
                # can return the mock if the policy is invalid?
                #
                # Answer: I think it would be cleaner if we would return an error
                # or both even
                if isinstance(final_results, TwinObject):
                    return final_results.private
                return final_results
            return state_result
        except Exception as e:
            return SyftError(message=f"Failed to run. {e}")


def get_outputs(context: AuthedServiceContext, output_history: OutputHistory) -> Any:
    # relative
    from .action_service import TwinMode

    if isinstance(output_history.outputs, list):
        if len(output_history.outputs) == 0:
            return None
        outputs = []
        for output_id in output_history.outputs:
            action_service = context.node.get_service("actionservice")
            result = action_service.get(
                context, uid=output_id, twin_mode=TwinMode.PRIVATE
            )
            if isinstance(result, OkErr):
                result = result.value
            outputs.append(result)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
    else:
        raise NotImplementedError


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

        if not isinstance(value, UID):
            raise Exception(f"Input {k} must have a UID not {type(v)}")
        filtered_kwargs[k] = value
    return filtered_kwargs


TYPE_TO_SERVICE[UserCode] = UserCodeService
SERVICE_TO_TYPES[UserCodeService].update({UserCode})
