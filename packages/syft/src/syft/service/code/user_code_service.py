# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from result import OkErr
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
from ..context import AuthedServiceContext
from ..policy.policy import OutputHistory
from ..request.request import SubmitRequest
from ..request.request import UserCodeStatusChange
from ..request.request_service import RequestService
from ..response import SyftError
from ..response import SyftNotReady
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .user_code import SubmitUserCode
from .user_code import UserCode
from .user_code import UserCodeStatus
from .user_code import load_approved_policy_code
from .user_code_stash import UserCodeStash


@instrument
@serializable()
class UserCodeService(AbstractService):
    store: DocumentStore
    stash: UserCodeStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = UserCodeStash(store=store)

    @service_method(path="code.submit", name="submit", roles=GUEST_ROLE_LEVEL)
    def submit(
        self, context: AuthedServiceContext, code: SubmitUserCode
    ) -> Union[UserCode, SyftError]:
        """Add User Code"""
        result = self.stash.set(context.credentials, code.to(UserCode, context=context))
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Submitted")

    def _request_code_execution(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode,
        reason: Optional[str] = "",
    ):
        user_code = code.to(UserCode, context=context)
        result = self.stash.set(context.credentials, user_code)
        if result.is_err():
            return SyftError(message=str(result.err()))

        linked_obj = LinkedObject.from_obj(user_code, node_uid=context.node.id)

        CODE_EXECUTE = UserCodeStatusChange(
            value=UserCodeStatus.EXECUTE, linked_obj=linked_obj
        )
        changes = [CODE_EXECUTE]

        request = SubmitRequest(changes=changes)
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request, reason=reason)

        # The Request service already returns either a SyftSuccess or SyftError
        return result

    @service_method(
        path="code.request_code_execution",
        name="request_code_execution",
        roles=GUEST_ROLE_LEVEL,
    )
    def request_code_execution(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode,
        reason: Optional[str] = "",
    ) -> Union[SyftSuccess, SyftError]:
        """Request Code execution on user code"""
        return self._request_code_execution(context=context, code=code, reason=reason)

    @service_method(path="code.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[UserCode], SyftError]:
        """Get a Dataset"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code.get_by_id", name="get_by_id")
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get a User Code Item"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            user_code = result.ok()
            if user_code and user_code.input_policy_state:
                # TODO replace with LinkedObject Context
                user_code.node_uid = context.node.id
            return user_code
        return SyftError(message=result.err())

    @service_method(path="code.get_all_for_user", name="get_all_for_user")
    def get_all_for_user(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        """Get All User Code Items for User's VerifyKey"""
        # TODO: replace with incoming user context and key
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def update_code_state(
        self, context: AuthedServiceContext, code_item: UserCode
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.update(context.credentials, code_item)
        if result.is_ok():
            return SyftSuccess(message="Code State Updated")
        return SyftError(message="Unable to Update Code State")

    def load_user_code(self, context: AuthedServiceContext) -> None:
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_ok():
            user_code_items = result.ok()
            load_approved_policy_code(user_code_items=user_code_items)

    @service_method(path="code.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        try:
            filtered_kwargs = filter_kwargs(kwargs)
            result = self.stash.get_by_uid(context.credentials, uid=uid)
            if not result.is_ok():
                return SyftError(message=result.err())

            # Unroll variables
            code_item = result.ok()
            code_status = code_item.status

            # Check if the user has permission to execute the code
            # They can execute if they are root user or if they are the user who submitted the code
            if not (
                context.credentials == context.node.verify_key
                or context.credentials == code_item.user_verify_key
            ):
                return SyftError(
                    message=f"Code Execution Permission: {context.credentials} denied"
                )

            # Check if the code is approved
            if code_status.for_context(context) != UserCodeStatus.EXECUTE:
                if code_status.for_context(context) == UserCodeStatus.SUBMITTED:
                    string = ""
                    for node_view, status in code_status.base_dict.items():
                        string += f"Code status on node '{node_view.node_name}' is '{status.value}'. "
                    return SyftNotReady(
                        message=f"{type(code_item)} Your code is waiting for approval. {string}"
                    )
                return SyftError(
                    message=f"{type(code_item)} Your code cannot be run: {code_status.for_context(context)}"
                )

            output_policy = code_item.output_policy
            if output_policy is None:
                raise Exception("Output policy not approved", code_item)

            # Check if the OutputPolicy is valid
            is_valid = output_policy.valid

            if not is_valid:
                if len(output_policy.output_history) > 0:
                    result = get_outputs(
                        context=context,
                        output_history=output_policy.output_history[-1],
                    )
                    return result.as_empty()
                return is_valid

            # Execute the code item
            action_service = context.node.get_service("actionservice")
            result: Result = action_service._user_code_execute(
                context, code_item, filtered_kwargs
            )
            if isinstance(result, str):
                return SyftError(message=result)

            # Apply Output Policy to the results and update the OutputPolicyState
            result: Union[ActionObject, TwinObject] = result.ok()
            output_policy.apply_output(context=context, outputs=result)
            code_item.output_policy = output_policy
            update_success = self.update_code_state(
                context=context, code_item=code_item
            )
            if not update_success:
                return update_success
            if isinstance(result, TwinObject):
                return result.mock
            else:
                return result.as_empty()
        except Exception as e:
            return SyftError(message=f"Failed to run. {e}")


def get_outputs(context: AuthedServiceContext, output_history: OutputHistory) -> Any:
    # relative
    from ...service.action.action_object import TwinMode

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
    from ...types.twin_object import TwinObject
    from ..action.action_object import ActionObject
    from ..dataset.dataset import Asset

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
