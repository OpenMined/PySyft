# stdlib
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Err
from result import Ok
from result import OkErr
from result import Result

# relative
from ...abstract_node import NodeType
from ...client.enclave_client import EnclaveClient
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..network.routes import route_to_connection
from ..request.request import Request
from ..request.request import SubmitRequest
from ..request.request import UserCodeStatusChange
from ..request.request_service import RequestService
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_roles import ServiceRole
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
        self, context: AuthedServiceContext, code: Union[UserCode, SubmitUserCode]
    ) -> Union[UserCode, SyftError]:
        """Add User Code"""
        result = self._submit(context=context, code=code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Submitted")

    def _submit(
        self, context: AuthedServiceContext, code: Union[UserCode, SubmitUserCode]
    ) -> Result:
        if not isinstance(code, UserCode):
            code = code.to(UserCode, context=context)
        result = self.stash.set(context.credentials, code)
        return result

    @service_method(
        path="code.sync_code_from_request",
        name="sync_code_from_request",
        roles=GUEST_ROLE_LEVEL,
    )
    def sync_code_from_request(
        self,
        context: AuthedServiceContext,
        request: Request,
    ) -> Union[SyftSuccess, SyftError]:
        """Re-submit request from a different node"""

        # This request is from a different node, ensure worker pool is not set
        code: UserCode = deepcopy(request.code)
        return self.submit(context=context, code=code)

    @service_method(
        path="code.get_by_service_func_name",
        name="get_by_service_func_name",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_by_service_name(
        self, context: AuthedServiceContext, service_func_name: str
    ):
        result = self.stash.get_by_service_func_name(
            context.credentials, service_func_name=service_func_name
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    def solve_nested_requests(self, context: AuthedServiceContext, code: UserCode):
        nested_requests = code.nested_requests
        nested_codes = {}
        for service_func_name, version in nested_requests.items():
            codes = self.get_by_service_name(
                context=context, service_func_name=service_func_name
            )
            if isinstance(codes, SyftError):
                return codes
            if version == "latest":
                nested_codes[service_func_name] = codes[-1]
            else:
                nested_codes[service_func_name] = codes[int(version)]

        return nested_codes

    def _request_code_execution(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode,
        reason: Optional[str] = "",
    ):
        user_code: UserCode = code.to(UserCode, context=context)
        return self._request_code_execution_inner(context, user_code, reason)

    def _request_code_execution_inner(
        self,
        context: AuthedServiceContext,
        user_code: UserCode,
        reason: Optional[str] = "",
    ):
        if not all(
            x in user_code.input_owner_verify_keys for x in user_code.output_readers
        ):
            raise ValueError("outputs can only be distributed to input owners")

        # check if the code with the same name and content already exists in the stash

        find_results = self.stash.get_by_code_hash(
            context.credentials, code_hash=user_code.code_hash
        )
        if find_results.is_err():
            return SyftError(message=str(find_results.err()))
        find_results = find_results.ok()

        if find_results is not None:
            return SyftError(
                message="The code to be submitted (name and content) already exists"
            )

        worker_pool_service = context.node.get_service("SyftWorkerPoolService")
        pool_result = worker_pool_service._get_worker_pool(
            context,
            pool_name=user_code.worker_pool_name,
        )

        if isinstance(pool_result, SyftError):
            return pool_result

        result = self.stash.set(context.credentials, user_code)
        if result.is_err():
            return SyftError(message=str(result.err()))

        # Create a code history
        code_history_service = context.node.get_service("codehistoryservice")
        result = code_history_service.submit_version(context=context, code=user_code)
        if isinstance(result, SyftError):
            return result

        # Users that have access to the output also have access to the code item
        self.stash.add_permissions(
            [
                ActionObjectPermission(user_code.id, ActionPermission.READ, x)
                for x in user_code.output_readers
            ]
        )

        linked_obj = LinkedObject.from_obj(user_code, node_uid=context.node.id)

        CODE_EXECUTE = UserCodeStatusChange(
            value=UserCodeStatus.APPROVED, linked_obj=linked_obj
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

    @service_method(
        path="code.get_by_id", name="get_by_id", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[UserCode, SyftError]:
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

    @service_method(path="code.get_results", name="get_results", roles=GUEST_ROLE_LEVEL)
    def get_results(
        self, context: AuthedServiceContext, inp: Union[UID, UserCode]
    ) -> Union[List[UserCode], SyftError]:
        uid = inp.id if isinstance(inp, UserCode) else inp
        code_result = self.stash.get_by_uid(context.credentials, uid=uid)

        if code_result.is_err():
            return SyftError(message=code_result.err())
        code = code_result.ok()

        if code.is_enclave_code:
            # if the current node is not the enclave
            if not context.node.node_type == NodeType.ENCLAVE:
                connection = route_to_connection(code.enclave_metadata.route)
                enclave_client = EnclaveClient(
                    connection=connection,
                    credentials=context.node.signing_key,
                )
                outputs = enclave_client.code.get_results(code.id)
                if isinstance(outputs, list):
                    for output in outputs:
                        output.syft_action_data  # noqa: B018
                else:
                    outputs.syft_action_data  # noqa: B018
                return outputs

            # if the current node is the enclave
            else:
                if not code.status.approved:
                    return code.status.get_status_message()

                if (output_policy := code.output_policy) is None:
                    return SyftError(message=f"Output policy not approved {code}")

                if len(output_policy.output_history) > 0:
                    return resolve_outputs(
                        context=context, output_ids=output_policy.last_output_ids
                    )
                else:
                    return SyftError(message="No results available")
        else:
            return SyftError(message="Endpoint only supported for enclave code")

    def is_execution_allowed(self, code, context, output_policy):
        if not code.status.approved:
            return code.status.get_status_message()
        # Check if the user has permission to execute the code.
        elif not (has_code_permission := self.has_code_permission(code, context)):
            return has_code_permission
        elif not code.output_policy_approved:
            return SyftError("Output policy not approved", code)
        elif not output_policy.valid:
            return output_policy.valid
        else:
            return True

    def is_execution_on_owned_args_allowed(self, context: AuthedServiceContext) -> bool:
        if context.role == ServiceRole.ADMIN:
            return True
        user_service = context.node.get_service("userservice")
        current_user = user_service.get_current_user(context=context)
        return current_user.mock_execution_permission

    def keep_owned_kwargs(
        self, kwargs: Dict[str, Any], context: AuthedServiceContext
    ) -> Dict[str, Any]:
        """Return only the kwargs that are owned by the user"""
        action_service = context.node.get_service("actionservice")

        mock_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, UID):
                # Jobs have UID kwargs instead of ActionObject
                v = action_service.get(context, uid=v)
                if v.is_ok():
                    v = v.ok()
            if (
                isinstance(v, ActionObject)
                and v.syft_client_verify_key == context.credentials
            ):
                mock_kwargs[k] = v
        return mock_kwargs

    def is_execution_on_owned_args(
        self, kwargs: Dict[str, Any], context: AuthedServiceContext
    ) -> bool:
        return len(self.keep_owned_kwargs(kwargs, context)) == len(kwargs)

    @service_method(path="code.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        kwargs.pop("result_id", None)
        result = self._call(context, uid, **kwargs)
        if result.is_err():
            return SyftError(message=result.err())
        else:
            return result.ok()

    def _call(
        self,
        context: AuthedServiceContext,
        uid: UID,
        result_id: Optional[UID] = None,
        **kwargs: Any,
    ) -> Result[ActionObject, Err]:
        """Call a User Code Function"""
        try:
            code_result = self.stash.get_by_uid(context.credentials, uid=uid)
            if code_result.is_err():
                return code_result
            code: UserCode = code_result.ok()

            # Set Permissions
            if self.is_execution_on_owned_args(kwargs, context):
                if self.is_execution_on_owned_args_allowed(context):
                    context.has_execute_permissions = True
                else:
                    return Err(
                        "You do not have the permissions for mock execution, please contact the admin"
                    )
            override_execution_permission = (
                context.has_execute_permissions or context.role == ServiceRole.ADMIN
            )
            # Override permissions bypasses the cache, since we do not check in/out policies
            skip_fill_cache = override_execution_permission
            # We do not read from output policy cache if there are mock arguments
            skip_read_cache = len(self.keep_owned_kwargs(kwargs, context)) > 0

            # Check output policy
            output_policy = code.output_policy
            if not override_execution_permission:
                can_execute = self.is_execution_allowed(
                    code=code, context=context, output_policy=output_policy
                )
                if not can_execute:
                    if not code.output_policy_approved:
                        return Err(
                            "Execution denied: Your code is waiting for approval"
                        )
                    if not (is_valid := output_policy.valid):
                        if (
                            len(output_policy.output_history) > 0
                            and not skip_read_cache
                        ):
                            result = resolve_outputs(
                                context=context,
                                output_ids=output_policy.last_output_ids,
                            )
                            return Ok(result.as_empty())
                        else:
                            return is_valid.to_result()
                    return can_execute.to_result()

            # Execute the code item
            action_service = context.node.get_service("actionservice")

            kwarg2id = map_kwargs_to_id(kwargs)
            result_action_object: Result[
                Union[ActionObject, TwinObject], str
            ] = action_service._user_code_execute(
                context, code, kwarg2id, result_id=result_id
            )
            if result_action_object.is_err():
                return result_action_object
            else:
                result_action_object = result_action_object.ok()

            output_result = action_service.set_result_to_store(
                result_action_object, context, code.output_policy
            )

            if output_result.is_err():
                return output_result
            result = output_result.ok()

            # Apply Output Policy to the results and update the OutputPolicyState

            # this currently only works for nested syft_functions
            # and admins executing on high side (TODO, decide if we want to increment counter)
            if not skip_fill_cache:
                output_policy.apply_output(context=context, outputs=result)
                code.output_policy = output_policy
                if not (
                    update_success := self.update_code_state(
                        context=context, code_item=code
                    )
                ):
                    return update_success.to_result()
            has_result_read_permission = context.extra_kwargs.get(
                "has_result_read_permission", False
            )
            if isinstance(result, TwinObject):
                if has_result_read_permission:
                    return Ok(result.private)
                else:
                    return Ok(result.mock)
            elif result.is_mock:
                return Ok(result)
            elif result.syft_action_data_type is Err:
                # result contains the error but the request was handled correctly
                return result.syft_action_data
            elif has_result_read_permission:
                return Ok(result)
            else:
                return Ok(result.as_empty())
        except Exception as e:
            # stdlib
            import traceback

            return Err(value=f"Failed to run. {e}, {traceback.format_exc()}")

    def has_code_permission(self, code_item, context):
        if not (
            context.credentials == context.node.verify_key
            or context.credentials == code_item.user_verify_key
        ):
            return SyftError(
                message=f"Code Execution Permission: {context.credentials} denied"
            )
        return SyftSuccess(message="you have permission")


def resolve_outputs(
    context: AuthedServiceContext,
    output_ids: Optional[Union[List[UID], Dict[str, UID]]],
) -> Any:
    # relative
    from ...service.action.action_object import TwinMode

    if isinstance(output_ids, list):
        if len(output_ids) == 0:
            return None
        outputs = []
        for output_id in output_ids:
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


def map_kwargs_to_id(kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
