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
from .request import UserCodeStatusChange
from .response import SyftError
from .response import SyftNotReady
from .response import SyftSuccess
from .serializable import serializable
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
from .service import service_method
from .uid import UID
from .user_code import OutputHistory
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
        result = self.stash.set(code.to(UserCode, context=context))
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Submitted")

    def _code_execution(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode,
    ):
        # relative
        from .request import SubmitRequest
        from .request_service import RequestService

        user_code = code.to(UserCode, context=context)
        result = self.stash.set(user_code)
        if result.is_err():
            return SyftError(message=str(result.err()))

        linked_obj = LinkedObject.from_obj(user_code, node_uid=context.node.id)

        CODE_EXECUTE = UserCodeStatusChange(
            value=UserCodeStatus.EXECUTE, linked_obj=linked_obj
        )
        request = SubmitRequest(changes=[CODE_EXECUTE])
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request)

        # The Request service already returns either a SyftSuccess or SyftError
        return result

    @service_method(path="code.request_code_execution", name="request_code_execution")
    def request_code_execution(
        self, context: AuthedServiceContext, code: SubmitUserCode
    ) -> Union[SyftSuccess, SyftError]:
        """Request Code execution on user code"""
        return self._code_execution(context=context, code=code)

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

    @service_method(path="code.call", name="call")
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        try:
            filtered_kwargs = filter_kwargs(kwargs)
            result = self.stash.get_by_uid(uid=uid)
            if result.is_ok():
                code_item = result.ok()
                if code_item.status.for_context(context) == UserCodeStatus.EXECUTE:
                    is_valid = code_item.output_policy_state.valid
                    if not is_valid:
                        if (
                            len(
                                code_item.output_policy_state.output_history,
                            )
                            > 0
                        ):
                            return get_outputs(
                                context=context,
                                output_history=code_item.output_policy_state.output_history[
                                    -1
                                ],
                            )
                        return is_valid
                    else:
                        action_service = context.node.get_service("actionservice")
                        result = action_service._user_code_execute(
                            context, code_item, filtered_kwargs
                        )
                        if isinstance(result, str):
                            return SyftError(message=result)
                        if result.is_ok():
                            result = result.ok()
                            code_item.output_policy_state.update_state(
                                context=context, outputs=result.id
                            )

                            state_result = self.update_code_state(
                                context=context, code_item=code_item
                            )

                            if state_result:
                                return result
                            else:
                                return state_result
                elif code_item.status.for_context(context) == UserCodeStatus.SUBMITTED:
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
