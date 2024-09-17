# stdlib
from enum import Enum
from typing import Any
from typing import TypeVar

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...store.linked_obj import LinkedObject
from ...types.errors import SyftException
from ...types.result import Err
from ...types.result import as_result
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..output.output_service import ExecutionOutput
from ..policy.policy import InputPolicyValidEnum
from ..policy.policy import OutputPolicy
from ..request.request import Request
from ..request.request import SubmitRequest
from ..request.request import SyncedUserCodeStatusChange
from ..request.request import UserCodeStatusChange
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
from .user_code import UserCodeUpdate
from .user_code import get_code_hash
from .user_code import load_approved_policy_code
from .user_code_stash import UserCodeStash


class HasCodePermissionEnum(str, Enum):
    ACCEPTED = "Has permission"
    DENIED = "Permission denied"


class IsExecutionAllowedEnum(str, Enum):
    ALLOWED = "Execution allowed"
    NO_PERMISSION = "Execution denied: You do not have permission to execute code"
    NOT_APPROVED = "Execution denied: Your code is waiting for approval"
    OUTPUT_POLICY_NONE = "Execution denied: Output policy is not set"
    INVALID_OUTPUT_POLICY = "Execution denied: Output policy is not valid"
    OUTPUT_POLICY_NOT_APPROVED = "Execution denied: Output policy not approved"


@serializable(canonical_name="UserCodeService", version=1)
class UserCodeService(AbstractService):
    stash: UserCodeStash

    def __init__(self, store: DBManager) -> None:
        self.stash = UserCodeStash(store=store)

    @service_method(
        path="code.submit",
        name="submit",
        roles=GUEST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def submit(
        self, context: AuthedServiceContext, code: SubmitUserCode
    ) -> SyftSuccess:
        """Add User Code"""
        user_code = self._submit(context, code, exists_ok=False).unwrap()
        return SyftSuccess(
            message="User Code Submitted", require_api_update=True, value=user_code
        )

    @as_result(SyftException)
    def _submit(
        self,
        context: AuthedServiceContext,
        submit_code: SubmitUserCode,
        exists_ok: bool = False,
    ) -> UserCode:
        """
        Submit a UserCode.

        If exists_ok is True, the function will return the existing code if it exists.

        Args:
            context (AuthedServiceContext): context
            submit_code (SubmitUserCode): UserCode to submit
            exists_ok (bool, optional): If True, return the existing code if it exists.
                If false, existing codes returns Err. Defaults to False.

        Returns:
            Result[UserCode, str]: New UserCode or error
        """
        try:
            existing_code = self.stash.get_by_code_hash(
                context.credentials,
                code_hash=get_code_hash(submit_code.code, context.credentials),
            ).unwrap()
            # no exception, code exists
            if exists_ok:
                return existing_code
            else:
                raise SyftException(
                    public_message="UserCode with this code already exists"
                )
        except NotFoundException:
            pass

        code = submit_code.to(UserCode, context=context)
        result = self._post_user_code_transform_ops(context, code)

        if result.is_err():
            # if the validation fails, we should remove the user code status
            # and code version to prevent dangling status
            root_context = AuthedServiceContext(
                credentials=context.server.verify_key, server=context.server
            )

            if code.status_link is not None:
                _ = context.server.services.user_code_status.remove(
                    root_context, code.status_link.object_uid
                )

            # result.unwrap() will raise any exceptions from post_user_code_transform_ops
            result.unwrap()

        return self.stash.set(context.credentials, code).unwrap()

    @service_method(
        path="code.update",
        name="update",
        roles=ADMIN_ROLE_LEVEL,
        autosplat=["code_update"],
        unwrap_on_success=False,
    )
    def update(
        self,
        context: AuthedServiceContext,
        code_update: UserCodeUpdate,
    ) -> SyftSuccess:
        updated_code = self.stash.update(context.credentials, code_update).unwrap()
        return SyftSuccess(message="UserCode updated successfully", value=updated_code)

    @service_method(
        path="code.delete",
        name="delete",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def delete(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        """Delete User Code"""
        self.stash.delete_by_uid(context.credentials, uid).unwrap()
        return SyftSuccess(message=f"User Code {uid} deleted", value=uid)

    @service_method(
        path="code.get_by_service_func_name",
        name="get_by_service_func_name",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_by_service_name(
        self, context: AuthedServiceContext, service_func_name: str
    ) -> list[UserCode]:
        return self.stash.get_by_service_func_name(
            context.credentials, service_func_name=service_func_name
        ).unwrap()

    # TODO: Add usercode errors
    @as_result(SyftException)
    def _post_user_code_transform_ops(
        self,
        context: AuthedServiceContext,
        user_code: UserCode,
    ) -> UserCode:
        if user_code.output_readers is None:
            raise SyftException(
                public_message=f"there is no verified output readers for {user_code}"
            )
        if user_code.input_owner_verify_keys is None:
            raise SyftException(
                public_message=f"there is no verified input owners for {user_code}"
            )
        if not all(
            x in user_code.input_owner_verify_keys for x in user_code.output_readers
        ):
            raise SyftException(
                public_message="outputs can only be distributed to input owners"
            )
        context.server.services.syft_worker_pool._get_worker_pool(
            context,
            pool_name=user_code.worker_pool_name,
        )

        # Create a code history
        context.server.services.code_history.submit_version(
            context=context, code=user_code
        )

        return user_code

    @as_result(SyftException)
    def _request_code_execution(
        self,
        context: AuthedServiceContext,
        user_code: UserCode,
        reason: str | None = "",
    ) -> Request:
        # Cannot make multiple requests for the same code
        # FIX: Change requestservice result type
        existing_requests = context.server.services.request.get_by_usercode_id(
            context, user_code.id
        )

        if len(existing_requests) > 0:
            raise SyftException(
                public_message=(
                    f"Request {existing_requests[0].id} already exists for this UserCode."
                    f" Please use the existing request, or submit a new UserCode to create a new request."
                )
            )

        # Users that have access to the output also have access to the code item
        if user_code.output_readers is not None:
            self.stash.add_permissions(
                [
                    ActionObjectPermission(user_code.id, ActionPermission.READ, x)
                    for x in user_code.output_readers
                ]
            )

        code_link = LinkedObject.from_obj(user_code, server_uid=context.server.id)

        # Requests made on low side are synced, and have their status computed instead of set manually.
        if user_code.is_l0_deployment:
            status_change = SyncedUserCodeStatusChange(
                value=UserCodeStatus.APPROVED,
                linked_obj=user_code.status_link,
                linked_user_code=code_link,
            )
        else:
            status_change = UserCodeStatusChange(
                value=UserCodeStatus.APPROVED,
                linked_obj=user_code.status_link,
                linked_user_code=code_link,
            )
        changes = [status_change]

        request = SubmitRequest(changes=changes)
        result = context.server.services.request.submit(
            context=context, request=request, reason=reason
        )

        return result

    @as_result(SyftException, NotFoundException, StashException)
    def _get_or_submit_user_code(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode | UserCode,
    ) -> UserCode:
        """
        - If the code is a UserCode, check if it exists and return
        - If the code is a SubmitUserCode and the same code hash exists, return the existing code
        - If the code is a SubmitUserCode and the code hash does not exist, submit the code
        """
        if isinstance(code, UserCode):
            return self.stash.get_by_uid(context.credentials, code.id).unwrap()
        else:  # code: SubmitUserCode
            # Submit new UserCode, or get existing UserCode with the same code hash
            # TODO: Why is this tagged as unreachable?
            return self._submit(context, code, exists_ok=True).unwrap()  # type: ignore[unreachable]

    @service_method(
        path="code.request_code_execution",
        name="request_code_execution",
        roles=GUEST_ROLE_LEVEL,
    )
    def request_code_execution(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode | UserCode,
        reason: str | None = "",
    ) -> Request:
        """Request Code execution on user code"""
        user_code = self._get_or_submit_user_code(context, code).unwrap()

        result = self._request_code_execution(
            context,
            user_code,
            reason,
        ).unwrap()

        return result

    @service_method(path="code.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> list[UserCode]:
        """Get a Dataset"""
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(
        path="code.get_by_id", name="get_by_id", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_by_uid(self, context: AuthedServiceContext, uid: UID) -> UserCode:
        """Get a User Code Item"""
        user_code = self.stash.get_by_uid(context.credentials, uid=uid).unwrap()
        if user_code and user_code.input_policy_state and context.server is not None:
            # TODO replace with LinkedObject Context
            user_code.server_uid = context.server.id
        return user_code

    @service_method(
        path="code.get_all_for_user",
        name="get_all_for_user",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_for_user(self, context: AuthedServiceContext) -> list[UserCode]:
        """Get All User Code Items for User's VerifyKey"""
        # TODO: replace with incoming user context and key
        return self.stash.get_all(context.credentials).unwrap()

    def update_code_state(
        self, context: AuthedServiceContext, code_item: UserCode
    ) -> UserCode:
        context = context.as_root_context()
        return self.stash.update(context.credentials, code_item).unwrap()

    @as_result(SyftException)
    def load_user_code(self, context: AuthedServiceContext) -> None:
        user_code_items = self.stash.get_all(credentials=context.credentials).unwrap()
        load_approved_policy_code(user_code_items=user_code_items, context=context)

    # FIX: Exceptions etc
    def is_execution_allowed(
        self,
        code: UserCode,
        context: AuthedServiceContext,
        output_policy: OutputPolicy | None,
    ) -> IsExecutionAllowedEnum:
        status = code.get_status(context).unwrap()
        if not status.get_is_approved(context):
            return IsExecutionAllowedEnum.NOT_APPROVED
        elif self.has_code_permission(code, context) is HasCodePermissionEnum.DENIED:
            # TODO: Check enum above
            return IsExecutionAllowedEnum.NO_PERMISSION
        elif not code.is_output_policy_approved(context):
            return IsExecutionAllowedEnum.OUTPUT_POLICY_NOT_APPROVED

        if output_policy is None:
            return IsExecutionAllowedEnum.OUTPUT_POLICY_NONE

        try:
            output_policy.is_valid(context)
        except Exception:
            return IsExecutionAllowedEnum.INVALID_OUTPUT_POLICY

        return IsExecutionAllowedEnum.ALLOWED

    def is_execution_on_owned_args_allowed(self, context: AuthedServiceContext) -> bool:
        if context.role == ServiceRole.ADMIN:
            return True
        current_user = context.server.services.user.get_current_user(context=context)
        return current_user.mock_execution_permission

    def keep_owned_kwargs(
        self, kwargs: dict[str, Any], context: AuthedServiceContext
    ) -> dict[str, Any]:
        """Return only the kwargs that are owned by the user"""
        mock_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, UID):
                # Jobs have UID kwargs instead of ActionObject
                try:
                    v = context.server.services.action.get(context, uid=v)
                except Exception:  # nosec: we are skipping when dont find it
                    pass
            if (
                isinstance(v, ActionObject)
                and v.syft_client_verify_key == context.credentials
            ):
                mock_kwargs[k] = v
        return mock_kwargs

    def is_execution_on_owned_args(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        passed_kwargs: dict[str, Any],
    ) -> bool:
        # Check if all kwargs are owned by the user
        all_kwargs_are_owned = len(
            self.keep_owned_kwargs(passed_kwargs, context)
        ) == len(passed_kwargs)

        if not all_kwargs_are_owned:
            return False

        # Check if the kwargs match the code signature
        try:
            code = self.stash.get_by_uid(context.credentials, user_code_id).unwrap()
        except SyftException:
            return False

        # Skip the datasite and context kwargs, they are passed by the backend
        code_kwargs = set(code.signature.parameters.keys()) - {"datasite", "context"}

        passed_kwarg_keys = set(passed_kwargs.keys())
        return passed_kwarg_keys == code_kwargs

    @service_method(path="code.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> ActionObject:
        """Call a User Code Function"""
        kwargs.pop("result_id", None)
        return self._call(context, uid, **kwargs).unwrap()

    def valid_worker_pool_for_context(
        self, context: AuthedServiceContext, user_code: UserCode
    ) -> bool:
        """This is a temporary fix that is needed until every function is always just ran as job"""
        # relative
        from ...server.server import get_default_worker_pool_name

        has_custom_worker_pool = (
            user_code.worker_pool_name is not None
        ) and user_code.worker_pool_name != get_default_worker_pool_name()
        if has_custom_worker_pool and context.is_blocking_api_call:
            return False
        else:
            return True

    @as_result(SyftException)
    def _call(
        self,
        context: AuthedServiceContext,
        uid: UID,
        result_id: UID | None = None,
        **kwargs: Any,
    ) -> ActionObject:
        """Call a User Code Function"""
        code: UserCode = self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

        # Set Permissions
        if self.is_execution_on_owned_args(context, uid, kwargs):
            if self.is_execution_on_owned_args_allowed(context):
                # handles the case: if we have 1 or more owned args and execution permission
                # handles the case: if we have 0 owned args and execution permission
                context.has_execute_permissions = True
            elif len(kwargs) == 0:
                # handles the case: if we have 0 owned args and execution permission
                pass
            else:
                raise SyftException(
                    public_message="You do not have the permissions for mock execution, please contact the admin"
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
        output_policy = code.get_output_policy(context)

        # Check output policy
        if not override_execution_permission:
            output_history = code.get_output_history(context=context).unwrap()

            is_execution_allowed = self.is_execution_allowed(
                code=code,
                context=context,
                output_policy=output_policy,
            )

            if (
                is_execution_allowed is not IsExecutionAllowedEnum.ALLOWED
                or context.is_l0_lowside
            ):
                # We check output policy only in l2 deployment.
                # code is from low side (L0 setup)
                status = code.get_status(context).unwrap()

                if (
                    context.server_allows_execution_for_ds
                    and not status.get_is_approved(context)
                ):
                    raise SyftException(
                        public_message=status.get_status_message_l2(context)
                    )

                output_policy_is_valid = False
                try:
                    if output_policy:
                        output_policy_is_valid = output_policy.is_valid(context)
                except SyftException:
                    pass

                # if you cant run it or the results are being sycned from l0
                # lets have a look at the output history and possibly return that
                if not output_policy_is_valid or code.is_l0_deployment:
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
                            inp_policy_validation = input_policy.is_valid(
                                context,
                                usr_input_kwargs=kwarg2id,
                            )

                            if not inp_policy_validation:
                                raise SyftException(
                                    # TODO: Print what's inside
                                    public_message=InputPolicyValidEnum.INVALID
                                )

                        outputs = resolve_outputs(
                            context=context,
                            output_ids=last_executed_output.output_ids,
                        ).unwrap()

                        if outputs:
                            outputs = delist_if_single(outputs)

                        if code.is_l2_deployment:
                            # Skip output policy warning in L0 setup;
                            # admin overrides policy checks.
                            output_policy_message = (
                                "Your result has been fetched from output_history, "
                                "because your OutputPolicy is no longer valid."
                            )
                            context.add_warning(output_policy_message)
                        return outputs  # type: ignore

                raise SyftException(public_message=is_execution_allowed.value)

        # Execute the code item
        if not self.valid_worker_pool_for_context(context, code):
            raise SyftException(
                public_message="You tried to run a syft function attached to a worker pool in blocking mode,"
                "which is currently not supported. Run your function with `blocking=False` to run"
                " as a job on your worker pool."
            )
        action_obj = context.server.services.action._user_code_execute(
            context, code, kwarg2id, result_id
        ).unwrap()

        result = context.server.services.action.set_result_to_store(
            action_obj, context, code.get_output_policy(context)
        ).unwrap()

        # Apply Output Policy to the results and update the OutputPolicyState

        # this currently only works for nested syft_functions
        # and admins executing on high side (TODO, decide if we want to increment counter)
        # always store_execution_output on l0 setup
        is_l0_request = context.role == ServiceRole.ADMIN and code.is_l0_deployment

        if not skip_fill_cache and output_policy is not None or is_l0_request:
            code.store_execution_output(
                context=context,
                outputs=result,
                job_id=context.job_id,
                input_ids=kwarg2id,
            ).unwrap()

        has_result_read_permission = context.extra_kwargs.get(
            "has_result_read_permission", False
        )

        # TODO: Just to fix the issue with the current implementation
        if context.role == ServiceRole.ADMIN:
            has_result_read_permission = True

        if isinstance(result, TwinObject):
            if has_result_read_permission:
                return result.private
            else:
                return result.mock
        elif result.is_mock:  # type: ignore[unreachable]
            return result  # type: ignore[return-value]
        # TODO: Check this part after error handling PR
        elif result.syft_action_data_type is Err:
            # result contains the error but the request was handled correctly
            return result
        elif has_result_read_permission:
            return result
        else:
            return result.as_empty()

    def has_code_permission(
        self, code_item: UserCode, context: AuthedServiceContext
    ) -> HasCodePermissionEnum:
        if not (
            context.credentials == context.server.verify_key
            or context.credentials == code_item.user_verify_key
        ):
            return HasCodePermissionEnum.DENIED
        return HasCodePermissionEnum.ACCEPTED

    @service_method(
        path="code.store_execution_output",
        name="store_execution_output",
        roles=GUEST_ROLE_LEVEL,
    )
    def store_execution_output(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        outputs: Any,
        input_ids: dict[str, UID] | None = None,
        job_id: UID | None = None,
    ) -> ExecutionOutput:
        code: UserCode = self.stash.get_by_uid(
            context.credentials, user_code_id
        ).unwrap()

        is_admin = context.role == ServiceRole.ADMIN

        if (
            not code.get_status(context).unwrap().get_is_approved(context)
            and not is_admin
        ):
            raise SyftException(public_message="This UserCode is not approved")

        return code.store_execution_output(
            context=context,
            outputs=outputs,
            job_id=job_id,
            input_ids=input_ids,
        ).unwrap()


@as_result(SyftException)
def resolve_outputs(
    context: AuthedServiceContext,
    output_ids: list[UID],
) -> list[ActionObject] | None:
    # relative
    from ...service.action.action_object import TwinMode

    if isinstance(output_ids, list):
        if len(output_ids) == 0:
            return None

        outputs = []
        for output_id in output_ids:
            if context.server is not None:
                output = context.server.services.action.get(
                    context, uid=output_id, twin_mode=TwinMode.PRIVATE
                )
                outputs.append(output)
        return outputs
    else:
        raise SyftException(public_message="Cannot resolve type of output_ids")


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
