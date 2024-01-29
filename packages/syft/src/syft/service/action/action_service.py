# stdlib
import importlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
import numpy as np
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...types.datetime import DateTime
from ...types.syft_object import SyftObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..blob_storage.service import BlobStorageService
from ..code.user_code import UserCode
from ..code.user_code import execute_byte_code
from ..context import AuthedServiceContext
from ..policy.policy import retrieve_from_db
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import UserLibConfigRegistry
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_roles import ServiceRole
from .action_object import Action
from .action_object import ActionObject
from .action_object import ActionObjectPointer
from .action_object import ActionType
from .action_object import AnyActionObject
from .action_object import TwinMode
from .action_permissions import ActionObjectPermission
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionPermission
from .action_store import ActionStore
from .action_types import action_type_for_type
from .numpy import NumpyArrayObject
from .pandas import PandasDataFrameObject  # noqa: F401
from .pandas import PandasSeriesObject  # noqa: F401


@serializable()
class ActionService(AbstractService):
    def __init__(self, store: ActionStore) -> None:
        self.store = store

    @service_method(path="action.np_array", name="np_array")
    def np_array(self, context: AuthedServiceContext, data: Any) -> Any:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        np_obj = NumpyArrayObject(
            dtype=data.dtype,
            shape=data.shape,
            syft_action_data_cache=data,
            syft_node_location=context.node.id,
            syft_client_verify_key=context.credentials,
        )
        blob_store_result = np_obj._save_to_blob_storage()
        if isinstance(blob_store_result, SyftError):
            return blob_store_result

        np_pointer = self._set(context, np_obj)
        return np_pointer

    @service_method(
        path="action.set",
        name="set",
        roles=GUEST_ROLE_LEVEL,
    )
    def set(
        self,
        context: AuthedServiceContext,
        action_object: Union[ActionObject, TwinObject],
    ) -> Result[ActionObject, str]:
        return self._set(context, action_object, has_result_read_permission=True)

    def _set(
        self,
        context: AuthedServiceContext,
        action_object: Union[ActionObject, TwinObject],
        has_result_read_permission: bool = False,
    ) -> Result[ActionObject, str]:
        """Save an object to the action store"""
        # 🟡 TODO 9: Create some kind of type checking / protocol for SyftSerializable

        if isinstance(action_object, ActionObject):
            action_object.syft_created_at = DateTime.now()
        else:
            action_object.private_obj.syft_created_at = DateTime.now()
            action_object.mock_obj.syft_created_at = DateTime.now()

        # If either context or argument is True, has_result_read_permission is True
        has_result_read_permission = (
            context.extra_kwargs.get("has_result_read_permission", False)
            or has_result_read_permission
        )

        result = self.store.set(
            uid=action_object.id,
            credentials=context.credentials,
            syft_object=action_object,
            has_result_read_permission=has_result_read_permission,
        )
        if result.is_ok():
            if isinstance(action_object, TwinObject):
                if has_result_read_permission:
                    action_object = action_object.private
                else:
                    action_object = action_object.mock
            action_object.syft_point_to(context.node.id)
            return Ok(action_object)
        return result.err()

    @service_method(
        path="action.is_resolved", name="is_resolved", roles=GUEST_ROLE_LEVEL
    )
    def is_resolved(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> Result[Ok[bool], Err[str]]:
        """Get an object from the action store"""
        # relative
        from .action_data_empty import ActionDataLink

        result = self._get(context, uid)
        if result.is_ok():
            obj = result.ok()
            if isinstance(obj.syft_action_data, ActionDataLink):
                result = self.resolve_links(
                    context, obj.syft_action_data.action_object_id.id
                )

                # Checking in case any error occurred
                if result.is_err():
                    return result

                return Ok(result.syft_resolved)

            # If it's a leaf but not resolved yet, return false
            elif not obj.syft_resolved:
                return Ok(False)

            # If it's not an action data link or non resolved (empty). It's resolved
            return Ok(True)

        # If it's not in the store or permission error, return the error
        return result

    @service_method(
        path="action.resolve_links", name="resolve_links", roles=GUEST_ROLE_LEVEL
    )
    def resolve_links(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
    ) -> Result[Ok[ActionObject], Err[str]]:
        """Get an object from the action store"""
        # relative
        from .action_data_empty import ActionDataLink

        result = self.store.get(uid=uid, credentials=context.credentials)
        # If user has permission to get the object / object exists
        if result.is_ok():
            obj = result.ok()

            # If it's not a leaf
            if isinstance(obj.syft_action_data, ActionDataLink):
                nested_result = self.resolve_links(
                    context, obj.syft_action_data.action_object_id.id, twin_mode
                )
                return nested_result

            # If it's a leaf
            return result

        return result

    @service_method(path="action.get", name="get", roles=GUEST_ROLE_LEVEL)
    def get(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
        resolve_nested: bool = True,
    ) -> Result[Ok[ActionObject], Err[str]]:
        """Get an object from the action store"""
        return self._get(context, uid, twin_mode, resolve_nested=resolve_nested)

    def _get(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
        has_permission=False,
        resolve_nested: bool = True,
    ) -> Result[ActionObject, str]:
        """Get an object from the action store"""
        # stdlib

        # relative
        from .action_data_empty import ActionDataLink

        result = self.store.get(
            uid=uid, credentials=context.credentials, has_permission=has_permission
        )
        if result.is_ok():
            obj: Union[TwinObject, ActionObject] = result.ok()
            obj._set_obj_location_(
                context.node.id,
                context.credentials,
            )
            # Resolve graph links
            if (
                not isinstance(obj, TwinObject)
                and resolve_nested
                and isinstance(obj.syft_action_data, ActionDataLink)
            ):
                if not self.is_resolved(
                    context, obj.syft_action_data.action_object_id.id
                ).ok():
                    return SyftError(message="This object is not resolved yet.")
                result = self.resolve_links(
                    context, obj.syft_action_data.action_object_id.id, twin_mode
                )
                return result
            if isinstance(obj, TwinObject):
                if twin_mode == TwinMode.PRIVATE:
                    obj = obj.private
                    obj.syft_point_to(context.node.id)
                elif twin_mode == TwinMode.MOCK:
                    obj = obj.mock
                    obj.syft_point_to(context.node.id)
                else:
                    obj.mock.syft_point_to(context.node.id)
                    obj.private.syft_point_to(context.node.id)
            return Ok(obj)
        else:
            return result

    @service_method(
        path="action.get_pointer", name="get_pointer", roles=GUEST_ROLE_LEVEL
    )
    def get_pointer(
        self, context: AuthedServiceContext, uid: UID
    ) -> Result[ActionObjectPointer, str]:
        """Get a pointer from the action store"""
        result = self.store.get_pointer(
            uid=uid, credentials=context.credentials, node_uid=context.node.id
        )
        if result.is_ok():
            obj = result.ok()
            obj._set_obj_location_(
                context.node.id,
                context.credentials,
            )
            return Ok(obj)
        return Err(result.err())

    @service_method(path="action.get_mock", name="get_mock", roles=GUEST_ROLE_LEVEL)
    def get_mock(
        self, context: AuthedServiceContext, uid: UID
    ) -> Result[SyftError, SyftObject]:
        """Get a pointer from the action store"""
        result = self.store.get_mock(uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    # not a public service endpoint
    def _user_code_execute(
        self,
        context: AuthedServiceContext,
        code_item: UserCode,
        kwargs: Dict[str, Any],
        result_id: Optional[UID] = None,
    ) -> Result[ActionObjectPointer, Err]:
        override_execution_permission = (
            context.has_execute_permissions or context.role == ServiceRole.ADMIN
        )

        if not override_execution_permission:
            input_policy = code_item.input_policy
            filtered_kwargs = input_policy.filter_kwargs(
                kwargs=kwargs, context=context, code_item_id=code_item.id
            )
            if isinstance(filtered_kwargs, SyftError) or filtered_kwargs.is_err():
                return filtered_kwargs
            filtered_kwargs = filtered_kwargs.ok()
        else:
            filtered_kwargs = retrieve_from_db(code_item.id, kwargs, context).ok()
        # update input policy to track any input state
        # code_item.input_policy = input_policy

        if not override_execution_permission:
            expected_input_kwargs = set()
            for _inp_kwarg in code_item.input_policy.inputs.values():
                keys = _inp_kwarg.keys()
                for k in keys:
                    if k not in kwargs:
                        return Err(
                            f"{code_item.service_func_name}() missing required keyword argument: '{k}'"
                        )
                expected_input_kwargs.update(keys)

            permitted_input_kwargs = list(filtered_kwargs.keys())
            not_approved_kwargs = set(expected_input_kwargs) - set(
                permitted_input_kwargs
            )
            if len(not_approved_kwargs) > 0:
                return Err(
                    f"Input arguments: {not_approved_kwargs} to the function are not approved yet."
                )

        has_twin_inputs = False

        real_kwargs = {}
        for key, kwarg_value in filtered_kwargs.items():
            if isinstance(kwarg_value, TwinObject):
                has_twin_inputs = True
            real_kwargs[key] = kwarg_value

        result_id = UID() if result_id is None else result_id

        try:
            if not has_twin_inputs:
                # no twins
                filtered_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.NONE
                )
                exec_result = execute_byte_code(code_item, filtered_kwargs, context)
                if isinstance(exec_result.result, ActionObject):
                    result_action_object = ActionObject.link(
                        result_id=result_id, pointer_id=exec_result.result.id
                    )
                else:
                    result_action_object = wrap_result(result_id, exec_result.result)
            else:
                # twins
                private_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.PRIVATE
                )
                private_exec_result = execute_byte_code(
                    code_item, private_kwargs, context
                )
                result_action_object_private = wrap_result(
                    result_id, private_exec_result.result
                )

                mock_kwargs = filter_twin_kwargs(real_kwargs, twin_mode=TwinMode.MOCK)
                # relative
                from .action_data_empty import ActionDataEmpty

                if any(isinstance(v, ActionDataEmpty) for v in mock_kwargs.values()):
                    mock_exec_result_obj = ActionDataEmpty()
                else:
                    mock_exec_result = execute_byte_code(
                        code_item, mock_kwargs, context
                    )
                    mock_exec_result_obj = mock_exec_result.result

                result_action_object_mock = wrap_result(result_id, mock_exec_result_obj)

                result_action_object = TwinObject(
                    id=result_id,
                    private_obj=result_action_object_private,
                    mock_obj=result_action_object_mock,
                )
        except Exception as e:
            # import traceback
            # return Err(f"_user_code_execute failed. {e} {traceback.format_exc()}")
            return Err(f"_user_code_execute failed. {e}")
        return Ok(result_action_object)

    def set_result_to_store(self, result_action_object, context, output_policy=None):
        result_id = result_action_object.id
        # result_blob_id = result_action_object.syft_blob_storage_entry_id

        if output_policy is not None:
            output_readers = (
                output_policy.output_readers
                if not context.has_execute_permissions
                else []
            )
        else:
            output_readers = []

        read_permission = ActionPermission.READ

        result_action_object._set_obj_location_(
            context.node.id,
            context.credentials,
        )
        blob_store_result = result_action_object._save_to_blob_storage()
        if isinstance(blob_store_result, SyftError):
            return blob_store_result

        # IMPORTANT: DO THIS ONLY AFTER ._save_to_blob_storage
        if isinstance(result_action_object, TwinObject):
            result_blob_id = result_action_object.private.syft_blob_storage_entry_id
        else:
            result_blob_id = result_action_object.syft_blob_storage_entry_id

        # pass permission information to the action store as extra kwargs
        context.extra_kwargs = {"has_result_read_permission": True}

        set_result = self._set(context, result_action_object)

        if set_result.is_err():
            return set_result

        blob_storage_service: BlobStorageService = context.node.get_service(
            BlobStorageService
        )

        def store_permission(x):
            return ActionObjectPermission(result_id, read_permission, x)

        def blob_permission(x):
            return ActionObjectPermission(result_blob_id, read_permission, x)

        if len(output_readers) > 0:
            store_permissions = [store_permission(x) for x in output_readers]
            self.store.add_permissions(store_permissions)

            blob_permissions = [blob_permission(x) for x in output_readers]
            blob_storage_service.stash.add_permissions(blob_permissions)

        return set_result

    def execute_plan(
        self, plan, context: AuthedServiceContext, plan_kwargs: Dict[str, ActionObject]
    ):
        id2inpkey = {v.id: k for k, v in plan.inputs.items()}

        for plan_action in plan.actions:
            if (
                hasattr(plan_action.remote_self, "id")
                and plan_action.remote_self.id in id2inpkey
            ):
                plan_action.remote_self = plan_kwargs[
                    id2inpkey[plan_action.remote_self.id]
                ]
            for i, arg in enumerate(plan_action.args):
                if arg in id2inpkey:
                    plan_action.args[i] = plan_kwargs[id2inpkey[arg]]

            for k, arg in enumerate(plan_action.kwargs):
                if arg in id2inpkey:
                    plan_action.kwargs[k] = plan_kwargs[id2inpkey[arg]]

        for plan_action in plan.actions:
            action_res = self.execute(context, plan_action)
            if isinstance(action_res, SyftError):
                return action_res
        result_id = plan.outputs[0].id
        return self._get(context, result_id, TwinMode.MOCK, has_permission=True)

    def call_function(self, context: AuthedServiceContext, action: Action):
        # run function/class init
        _user_lib_config_registry = UserLibConfigRegistry.from_user(context.credentials)
        absolute_path = f"{action.path}.{action.op}"
        if absolute_path in _user_lib_config_registry:
            # TODO: implement properly
            # Now we are assuming its a function/class
            return execute_callable(self, context, action)
        else:
            return Err(
                f"Failed executing {action}. You have no permission for {absolute_path}"
            )

    def set_attribute(
        self,
        context: AuthedServiceContext,
        action: Action,
        resolved_self: Union[ActionObject, TwinObject],
    ):
        args, _ = resolve_action_args(action, context, self)
        if args.is_err():
            return Err(
                f"Failed executing action {action}, could not resolve args: {args.err()}"
            )
        else:
            args = args.ok()
        if not isinstance(args[0], ActionObject):
            return Err(
                f"Failed executing action {action} setattribute requires a non-twin string as first argument"
            )
        name = args[0].syft_action_data
        # dont do the whole filtering dance with the name
        args = [args[1]]

        if isinstance(resolved_self, TwinObject):
            # todo, create copy?
            private_args = filter_twin_args(args, twin_mode=TwinMode.PRIVATE)
            private_val = private_args[0]
            setattr(resolved_self.private.syft_action_data, name, private_val)
            # todo: what do we use as data for the mock here?
            # depending on permisisons?
            public_args = filter_twin_args(args, twin_mode=TwinMode.MOCK)
            public_val = public_args[0]
            setattr(resolved_self.mock.syft_action_data, name, public_val)
            return Ok(
                TwinObject(
                    id=action.result_id,
                    private_obj=ActionObject.from_obj(
                        resolved_self.private.syft_action_data
                    ),
                    private_obj_id=action.result_id,
                    mock_obj=ActionObject.from_obj(resolved_self.mock.syft_action_data),
                    mock_obj_id=action.result_id,
                )
            )
        else:
            # TODO: Implement for twinobject args
            args = filter_twin_args(args, twin_mode=TwinMode.NONE)
            val = args[0]
            setattr(resolved_self.syft_action_data, name, val)
            return Ok(
                ActionObject.from_obj(resolved_self.syft_action_data),
            )
            # todo: permissions
            # setattr(resolved_self.syft_action_data, name, val)
            # val = resolved_self.syft_action_data
            # result_action_object = Ok(wrap_result(action.result_id, val))

    def get_attribute(
        self, action: Action, resolved_self: Union[ActionObject, TwinObject]
    ):
        if isinstance(resolved_self, TwinObject):
            private_result = getattr(resolved_self.private.syft_action_data, action.op)
            mock_result = getattr(resolved_self.mock.syft_action_data, action.op)
            return Ok(
                TwinObject(
                    id=action.result_id,
                    private_obj=ActionObject.from_obj(private_result),
                    private_obj_id=action.result_id,
                    mock_obj=ActionObject.from_obj(mock_result),
                    mock_obj_id=action.result_id,
                )
            )
        else:
            val = getattr(resolved_self.syft_action_data, action.op)
            return Ok(wrap_result(action.result_id, val))

    def call_method(
        self,
        context: AuthedServiceContext,
        action: Action,
        resolved_self: Union[ActionObject, TwinObject],
    ):
        if isinstance(resolved_self, TwinObject):
            # method
            private_result = execute_object(
                self,
                context,
                resolved_self.private,
                action,
                twin_mode=TwinMode.PRIVATE,
            )
            if private_result.is_err():
                return Err(
                    f"Failed executing action {action}, result is an error: {private_result.err()}"
                )
            mock_result = execute_object(
                self, context, resolved_self.mock, action, twin_mode=TwinMode.MOCK
            )
            if mock_result.is_err():
                return Err(
                    f"Failed executing action {action}, result is an error: {mock_result.err()}"
                )

            private_result = private_result.ok()
            mock_result = mock_result.ok()

            return Ok(
                TwinObject(
                    id=action.result_id,
                    private_obj=private_result,
                    private_obj_id=action.result_id,
                    mock_obj=mock_result,
                    mock_obj_id=action.result_id,
                )
            )
        else:
            return execute_object(self, context, resolved_self, action)

    @service_method(path="action.execute", name="execute", roles=GUEST_ROLE_LEVEL)
    def execute(
        self, context: AuthedServiceContext, action: Action
    ) -> Result[ActionObject, Err]:
        """Execute an operation on objects in the action store"""
        # relative
        from .plan import Plan

        if action.action_type == ActionType.CREATEOBJECT:
            result_action_object = Ok(action.create_object)
            # print(action.create_object, "already in blob storage")
        elif action.action_type == ActionType.SYFTFUNCTION:
            usercode_service = context.node.get_service("usercodeservice")
            kwarg_ids = {}
            for k, v in action.kwargs.items():
                # transform lineage ids into ids
                kwarg_ids[k] = v.id
            result_action_object: Result[ActionObject, Err] = usercode_service._call(
                context, action.user_code_id, action.result_id, **kwarg_ids
            )
            return result_action_object
        elif action.action_type == ActionType.FUNCTION:
            result_action_object = self.call_function(context, action)
        else:
            resolved_self = self._get(
                context=context,
                uid=action.remote_self,
                twin_mode=TwinMode.NONE,
                has_permission=True,
            )
            if resolved_self.is_err():
                return Err(
                    f"Failed executing action {action}, could not resolve self: {resolved_self.err()}"
                )
            resolved_self = resolved_self.ok()
            if action.op == "__call__" and resolved_self.syft_action_data_type == Plan:
                result_action_object = self.execute_plan(
                    plan=resolved_self.syft_action_data,
                    context=context,
                    plan_kwargs=action.kwargs,
                )
                return result_action_object
            elif action.action_type == ActionType.SETATTRIBUTE:
                result_action_object = self.set_attribute(
                    context, action, resolved_self
                )
            elif action.action_type == ActionType.GETATTRIBUTE:
                result_action_object = self.get_attribute(action, resolved_self)
            elif action.action_type == ActionType.METHOD:
                result_action_object = self.call_method(context, action, resolved_self)
            else:
                return Err("Unknown action")

        if result_action_object.is_err():
            return Err(
                f"Failed executing action {action}, result is an error: {result_action_object.err()}"
            )
        else:
            result_action_object = result_action_object.ok()

        # check if we have read permissions on the result
        has_result_read_permission = self.has_read_permission_for_action_result(
            context, action
        )

        result_action_object._set_obj_location_(
            context.node.id,
            context.credentials,
        )

        blob_store_result = result_action_object._save_to_blob_storage()
        if isinstance(blob_store_result, SyftError):
            return blob_store_result

        # pass permission information to the action store as extra kwargs
        context.extra_kwargs = {
            "has_result_read_permission": has_result_read_permission
        }

        set_result = self._set(context, result_action_object)
        if set_result.is_err():
            return Err(
                f"Failed executing action {action}, set result is an error: {set_result.err()}"
            )

        return set_result

    def has_read_permission_for_action_result(
        self, context: AuthedServiceContext, action: Action
    ) -> bool:
        action_obj_ids = (
            action.args + list(action.kwargs.values()) + [action.remote_self]
        )
        permissions = [
            ActionObjectREAD(uid=_id, credentials=context.credentials)
            for _id in action_obj_ids
        ]
        return self.store.has_permissions(permissions)

    @service_method(path="action.exists", name="exists", roles=GUEST_ROLE_LEVEL)
    def exists(
        self, context: AuthedServiceContext, obj_id: UID
    ) -> Result[SyftSuccess, SyftError]:
        """Checks if the given object id exists in the Action Store"""
        if self.store.exists(obj_id):
            return SyftSuccess(message=f"Object: {obj_id} exists")
        else:
            return SyftError(message=f"Object: {obj_id} does not exist")


def resolve_action_args(
    action: Action, context: AuthedServiceContext, service: ActionService
):
    has_twin_inputs = False
    args = []
    for arg_id in action.args:
        arg_value = service._get(
            context=context, uid=arg_id, twin_mode=TwinMode.NONE, has_permission=True
        )
        if arg_value.is_err():
            return arg_value, False
        if isinstance(arg_value.ok(), TwinObject):
            has_twin_inputs = True
        args.append(arg_value.ok())
    return Ok(args), has_twin_inputs


def resolve_action_kwargs(
    action: Action, context: AuthedServiceContext, service: ActionService
):
    has_twin_inputs = False
    kwargs = {}
    for key, arg_id in action.kwargs.items():
        kwarg_value = service._get(
            context=context, uid=arg_id, twin_mode=TwinMode.NONE, has_permission=True
        )
        if kwarg_value.is_err():
            return kwarg_value, False
        if isinstance(kwarg_value.ok(), TwinObject):
            has_twin_inputs = True
        kwargs[key] = kwarg_value.ok()
    return Ok(kwargs), has_twin_inputs


def execute_callable(
    service: ActionService,
    context: AuthedServiceContext,
    action: Action,
) -> Result[ActionObject, str]:
    args, has_arg_twins = resolve_action_args(action, context, service)
    kwargs, has_kwargs_twins = resolve_action_kwargs(action, context, service)
    has_twin_inputs = has_arg_twins or has_kwargs_twins
    if args.is_err():
        return args
    else:
        args = args.ok()
    if kwargs.is_err():
        return kwargs
    else:
        kwargs = kwargs.ok()

    # 🔵 TODO 10: Get proper code From old RunClassMethodAction to ensure the function
    # is not bound to the original object or mutated
    # stdlib

    # TODO: get from CMPTree is probably safer
    def _get_target_callable(path: str, op: str):
        path_elements = path.split(".")
        res = importlib.import_module(path_elements[0])
        for p in path_elements[1:]:
            res = getattr(res, p)
        res = getattr(res, op)
        return res

    target_callable = _get_target_callable(action.path, action.op)

    result = None
    try:
        if target_callable:
            if not has_twin_inputs:
                # if twin_mode == TwinMode.NONE and not has_twin_inputs:
                twin_mode = TwinMode.NONE
                # no twins
                filtered_args = filter_twin_args(args, twin_mode=twin_mode)
                filtered_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                result = target_callable(*filtered_args, **filtered_kwargs)
                result_action_object = wrap_result(action.result_id, result)
            else:
                twin_mode = TwinMode.PRIVATE
                private_args = filter_twin_args(args, twin_mode=twin_mode)
                private_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                private_result = target_callable(*private_args, **private_kwargs)
                result_action_object_private = wrap_result(
                    action.result_id, private_result
                )

                twin_mode = TwinMode.MOCK
                mock_args = filter_twin_args(args, twin_mode=twin_mode)
                mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                mock_result = target_callable(*mock_args, **mock_kwargs)
                result_action_object_mock = wrap_result(action.result_id, mock_result)

                result_action_object = TwinObject(
                    id=action.result_id,
                    private_obj=result_action_object_private,
                    mock_obj=result_action_object_mock,
                )

    except Exception as e:
        print("what is this exception", e)
        return Err(e)
    return Ok(result_action_object)


def execute_object(
    service: ActionService,
    context: AuthedServiceContext,
    resolved_self: ActionObject,
    action: Action,
    twin_mode: TwinMode = TwinMode.NONE,
) -> Result[Ok[Union[TwinObject, ActionObject]], Err[str]]:
    unboxed_resolved_self = resolved_self.syft_action_data
    _args, has_arg_twins = resolve_action_args(action, context, service)

    kwargs, has_kwargs_twins = resolve_action_kwargs(action, context, service)
    if _args.is_err():
        return _args
    else:
        args = _args.ok()
    if kwargs.is_err():
        return kwargs
    else:
        kwargs = kwargs.ok()
    has_twin_inputs = has_arg_twins or has_kwargs_twins

    # 🔵 TODO 10: Get proper code From old RunClassMethodAction to ensure the function
    # is not bound to the original object or mutated
    target_method = getattr(unboxed_resolved_self, action.op, None)
    result = None
    try:
        if target_method:
            if twin_mode == TwinMode.NONE and not has_twin_inputs:
                # no twins
                filtered_args = filter_twin_args(args, twin_mode=twin_mode)
                filtered_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                result = target_method(*filtered_args, **filtered_kwargs)
                result_action_object = wrap_result(action.result_id, result)
            elif twin_mode == TwinMode.NONE and has_twin_inputs:
                # self isn't a twin but one of the inputs is
                private_args = filter_twin_args(args, twin_mode=TwinMode.PRIVATE)
                private_kwargs = filter_twin_kwargs(kwargs, twin_mode=TwinMode.PRIVATE)
                private_result = target_method(*private_args, **private_kwargs)
                result_action_object_private = wrap_result(
                    action.result_id, private_result
                )

                mock_args = filter_twin_args(args, twin_mode=TwinMode.MOCK)
                mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=TwinMode.MOCK)
                mock_result = target_method(*mock_args, **mock_kwargs)
                result_action_object_mock = wrap_result(action.result_id, mock_result)

                result_action_object = TwinObject(
                    id=action.result_id,
                    private_obj=result_action_object_private,
                    mock_obj=result_action_object_mock,
                )
            elif twin_mode == twin_mode.PRIVATE:  # type: ignore
                # twin private path
                private_args = filter_twin_args(args, twin_mode=twin_mode)
                private_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                result = target_method(*private_args, **private_kwargs)
                result_action_object = wrap_result(action.result_id, result)
            elif twin_mode == twin_mode.MOCK:  # type: ignore
                # twin mock path
                mock_args = filter_twin_args(args, twin_mode=twin_mode)
                mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                target_method = getattr(unboxed_resolved_self, action.op, None)
                result = target_method(*mock_args, **mock_kwargs)
                result_action_object = wrap_result(action.result_id, result)
            else:
                raise Exception(
                    f"Bad combination of: twin_mode: {twin_mode} and has_twin_inputs: {has_twin_inputs}"
                )
        else:
            return Err("Missing target method")

    except Exception as e:
        return Err(e)

    return Ok(result_action_object)


def wrap_result(result_id: UID, result: Any) -> ActionObject:
    # 🟡 TODO 11: Figure out how we want to store action object results
    action_type = action_type_for_type(result)
    result_action_object = action_type(id=result_id, syft_action_data_cache=result)
    return result_action_object


def filter_twin_args(args: List[Any], twin_mode: TwinMode) -> Any:
    filtered = []
    for arg in args:
        if isinstance(arg, TwinObject):
            if twin_mode == TwinMode.PRIVATE:
                filtered.append(arg.private.syft_action_data)
            elif twin_mode == TwinMode.MOCK:
                filtered.append(arg.mock.syft_action_data)
            else:
                raise Exception(
                    f"Filter can only use {TwinMode.PRIVATE} or {TwinMode.MOCK}"
                )
        else:
            filtered.append(arg.syft_action_data)
    return filtered


def filter_twin_kwargs(kwargs: Dict, twin_mode: TwinMode) -> Any:
    filtered = {}
    for k, v in kwargs.items():
        if isinstance(v, TwinObject):
            if twin_mode == TwinMode.PRIVATE:
                filtered[k] = v.private.syft_action_data
            elif twin_mode == TwinMode.MOCK:
                filtered[k] = v.mock.syft_action_data
            else:
                raise Exception(
                    f"Filter can only use {TwinMode.PRIVATE} or {TwinMode.MOCK}"
                )
        else:
            filtered[k] = v.syft_action_data
    return filtered


TYPE_TO_SERVICE[ActionObject] = ActionService
TYPE_TO_SERVICE[TwinObject] = ActionService
TYPE_TO_SERVICE[AnyActionObject] = ActionService

SERVICE_TO_TYPES[ActionService].update({ActionObject, TwinObject, AnyActionObject})
