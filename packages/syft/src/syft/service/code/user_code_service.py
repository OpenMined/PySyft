# stdlib
from typing import Any
from typing import TypeVar
from typing import cast

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_node import NodeType
from ...client.enclave_client import EnclaveClient
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.cache_object import CachedSyftObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..network.routes import route_to_connection
from ..output.output_service import ExecutionOutput
from ..policy.policy import OutputPolicy
from ..request.request import Request
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
from ..user.user_roles import ADMIN_ROLE_LEVEL
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
        self, context: AuthedServiceContext, code: UserCode | SubmitUserCode
    ) -> UserCode | SyftError:
        """Add User Code"""
        result = self._submit(context=context, code=code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Submitted", require_api_update=True)

    def _submit(
        self, context: AuthedServiceContext, code: UserCode | SubmitUserCode
    ) -> Result[UserCode, str]:
        if not isinstance(code, UserCode):
            code = code.to(UserCode, context=context)  # type: ignore[unreachable]

        result = self.stash.set(context.credentials, code)
        return result

    @service_method(path="code.delete", name="delete", roles=ADMIN_ROLE_LEVEL)
    def delete(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        """Delete User Code"""
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Deleted")

    @service_method(
        path="code.get_by_service_func_name",
        name="get_by_service_func_name",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_by_service_name(
        self, context: AuthedServiceContext, service_func_name: str
    ) -> list[UserCode] | SyftError:
        result = self.stash.get_by_service_func_name(
            context.credentials, service_func_name=service_func_name
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    def _request_code_execution(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode,
        reason: str | None = "",
    ) -> Request | SyftError:
        user_code: UserCode = code.to(UserCode, context=context)
        result = self._validate_request_code_execution(context, user_code)
        if isinstance(result, SyftError):
            # if the validation fails, we should remove the user code status
            # and code version to prevent dangling status
            root_context = AuthedServiceContext(
                credentials=context.node.verify_key, node=context.node
            )
            _ = context.node.get_service("usercodestatusservice").remove(
                root_context, user_code.status_link.object_uid
            )
            return result
        result = self._request_code_execution_inner(context, user_code, reason)
        return result

    def _validate_request_code_execution(
        self,
        context: AuthedServiceContext,
        user_code: UserCode,
    ) -> SyftSuccess | SyftError:
        if user_code.output_readers is None:
            return SyftError(
                message=f"there is no verified output readers for {user_code}"
            )
        if user_code.input_owner_verify_keys is None:
            return SyftError(
                message=f"there is no verified input owners for {user_code}"
            )
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

        return SyftSuccess(message="")

    def _request_code_execution_inner(
        self,
        context: AuthedServiceContext,
        user_code: UserCode,
        reason: str | None = "",
    ) -> Request | SyftError:
        # Users that have access to the output also have access to the code item
        if user_code.output_readers is not None:
            self.stash.add_permissions(
                [
                    ActionObjectPermission(user_code.id, ActionPermission.READ, x)
                    for x in user_code.output_readers
                ]
            )

        code_link = LinkedObject.from_obj(user_code, node_uid=context.node.id)

        CODE_EXECUTE = UserCodeStatusChange(
            value=UserCodeStatus.APPROVED,
            linked_obj=user_code.status_link,
            linked_user_code=code_link,
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
        reason: str | None = "",
    ) -> SyftSuccess | SyftError:
        """Request Code execution on user code"""
        return self._request_code_execution(context=context, code=code, reason=reason)

    @service_method(path="code.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> list[UserCode] | SyftError:
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
    ) -> UserCode | SyftError:
        """Get a User Code Item"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            user_code = result.ok()
            if user_code and user_code.input_policy_state and context.node is not None:
                # TODO replace with LinkedObject Context
                user_code.node_uid = context.node.id
            return user_code
        return SyftError(message=result.err())

    @service_method(
        path="code.get_all_for_user",
        name="get_all_for_user",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_for_user(
        self, context: AuthedServiceContext
    ) -> SyftSuccess | SyftError:
        """Get All User Code Items for User's VerifyKey"""
        # TODO: replace with incoming user context and key
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def update_code_state(
        self, context: AuthedServiceContext, code_item: UserCode
    ) -> SyftSuccess | SyftError:
        context = context.as_root_context()
        result = self.stash.update(context.credentials, code_item)
        if result.is_ok():
            return SyftSuccess(message="Code State Updated")
        return SyftError(message="Unable to Update Code State")

    def load_user_code(self, context: AuthedServiceContext) -> None:
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_ok():
            user_code_items = result.ok()
            load_approved_policy_code(user_code_items=user_code_items, context=context)

    @service_method(path="code.get_results", name="get_results", roles=GUEST_ROLE_LEVEL)
    def get_results(
        self, context: AuthedServiceContext, inp: UID | UserCode
    ) -> list[UserCode] | SyftError:
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
                if enclave_client.code is None:
                    return SyftError(
                        message=f"{enclave_client} can't access the user code api"
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
                if not code.get_status(context.as_root_context()).approved:
                    return code.status.get_status_message()

                output_history = code.get_output_history(
                    context=context.as_root_context()
                )
                if isinstance(output_history, SyftError):
                    return output_history

                if len(output_history) > 0:
                    res = resolve_outputs(
                        context=context,
                        output_ids=output_history[-1].output_ids,
                    )
                    if res.is_err():
                        return res
                    res = delist_if_single(res.ok())
                    return Ok(res)
                else:
                    return SyftError(message="No results available")
        else:
            return SyftError(message="Endpoint only supported for enclave code")

    def is_execution_allowed(
        self,
        code: UserCode,
        context: AuthedServiceContext,
        output_policy: OutputPolicy | None,
    ) -> bool | SyftSuccess | SyftError | SyftNotReady:
        if not code.get_status(context).approved:
            return code.status.get_status_message()
        # Check if the user has permission to execute the code.
        elif not (has_code_permission := self.has_code_permission(code, context)):
            return has_code_permission
        elif not code.is_output_policy_approved(context):
            return SyftError("Output policy not approved", code)

        policy_is_valid = output_policy is not None and output_policy._is_valid(context)
        if not policy_is_valid:
            return policy_is_valid
        else:
            return True

    def is_execution_on_owned_args_allowed(
        self, context: AuthedServiceContext
    ) -> bool | SyftError:
        if context.role == ServiceRole.ADMIN:
            return True

        user_service = context.node.get_service("userservice")
        current_user = user_service.get_current_user(context=context)
        return current_user.mock_execution_permission

    def keep_owned_kwargs(
        self, kwargs: dict[str, Any], context: AuthedServiceContext
    ) -> dict[str, Any] | SyftError:
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
        self, kwargs: dict[str, Any], context: AuthedServiceContext
    ) -> bool:
        return len(self.keep_owned_kwargs(kwargs, context)) == len(kwargs)

    @service_method(path="code.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> CachedSyftObject | ActionObject | SyftSuccess | SyftError:
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
        result_id: UID | None = None,
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

            # Extract ids from kwargs
            kwarg2id = map_kwargs_to_id(kwargs)

            input_policy = code.get_input_policy(context)

            # Check output policy
            output_policy = code.get_output_policy(context)
            if not override_execution_permission:
                output_history = code.get_output_history(context=context)
                if isinstance(output_history, SyftError):
                    return Err(output_history.message)
                can_execute = self.is_execution_allowed(
                    code=code,
                    context=context,
                    output_policy=output_policy,
                )
                if not can_execute:
                    if not code.is_output_policy_approved(context):
                        return Err(
                            "Execution denied: Your code is waiting for approval"
                        )
                    if not (is_valid := output_policy._is_valid(context)):  # type: ignore
                        if len(output_history) > 0 and not skip_read_cache:
                            last_executed_output = output_history[-1]
                            # Check if the inputs of the last executed output match
                            # against the current input
                            if (
                                input_policy is not None
                                and not last_executed_output.check_input_ids(
                                    kwargs=kwarg2id
                                )
                            ):
                                inp_policy_validation = input_policy._is_valid(
                                    context,
                                    usr_input_kwargs=kwarg2id,
                                    code_item_id=code.id,
                                )
                                if inp_policy_validation.is_err():
                                    return inp_policy_validation

                            result: Result[ActionObject, str] = resolve_outputs(
                                context=context,
                                output_ids=last_executed_output.output_ids,
                            )
                            if result.is_err():
                                return result

                            res = delist_if_single(result.ok())
                            return Ok(
                                CachedSyftObject(
                                    result=res,
                                    error_msg=is_valid.message,
                                )
                            )
                        else:
                            return cast(Err, is_valid.to_result())
                    return can_execute.to_result()  # type: ignore

            # Execute the code item

            action_service = context.node.get_service("actionservice")

            result_action_object: Result[ActionObject | TwinObject, str] = (
                action_service._user_code_execute(
                    context, code, kwarg2id, result_id=result_id
                )
            )
            if result_action_object.is_err():
                return result_action_object
            else:
                result_action_object = result_action_object.ok()

            output_result = action_service.set_result_to_store(
                result_action_object, context, code.get_output_policy(context)
            )

            if output_result.is_err():
                return output_result
            result = output_result.ok()

            # Apply Output Policy to the results and update the OutputPolicyState

            # this currently only works for nested syft_functions
            # and admins executing on high side (TODO, decide if we want to increment counter)
            if not skip_fill_cache and output_policy is not None:
                res = code.store_as_history(
                    context=context,
                    outputs=result,
                    job_id=context.job_id,
                    input_ids=kwarg2id,
                )
                if isinstance(res, SyftError):
                    return Err(res.message)

            # output_policy.update_policy(context, result)
            # code.output_policy = output_policy
            # res = self.update_code_state(context, code)
            # print(res)

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

    def has_code_permission(
        self, code_item: UserCode, context: AuthedServiceContext
    ) -> SyftSuccess | SyftError:
        if not (
            context.credentials == context.node.verify_key
            or context.credentials == code_item.user_verify_key
        ):
            return SyftError(
                message=f"Code Execution Permission: {context.credentials} denied"
            )
        return SyftSuccess(message="you have permission")

    @service_method(
        path="code.store_as_history", name="store_as_history", roles=GUEST_ROLE_LEVEL
    )
    def store_as_history(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        outputs: Any,
        input_ids: dict[str, UID] | None = None,
        job_id: UID | None = None,
    ) -> ExecutionOutput | SyftError:
        code_result = self.stash.get_by_uid(context.credentials, user_code_id)
        if code_result.is_err():
            return SyftError(message=code_result.err())

        code: UserCode = code_result.ok()
        if not code.get_status(context).approved:
            return SyftError(message="Code is not approved")

        res = code.store_as_history(
            context=context,
            outputs=outputs,
            job_id=job_id,
            input_ids=input_ids,
        )
        return res


def resolve_outputs(
    context: AuthedServiceContext,
    output_ids: list[UID],
) -> Result[list[ActionObject], str]:
    # relative
    from ...service.action.action_object import TwinMode

    if isinstance(output_ids, list):
        if len(output_ids) == 0:
            return None
        outputs = []
        for output_id in output_ids:
            if context.node is not None:
                action_service = context.node.get_service("actionservice")
                result = action_service.get(
                    context, uid=output_id, twin_mode=TwinMode.PRIVATE
                )
                if result.is_err():
                    return result
                outputs.append(result.ok())
        return Ok(outputs)
    else:
        raise NotImplementedError


T = TypeVar("T")


def delist_if_single(result: list[T]) -> T | list[T]:
    if len(result) == 1:
        return result[0]
    return result


def map_kwargs_to_id(kwargs: dict[str, Any]) -> dict[str, Any]:
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
