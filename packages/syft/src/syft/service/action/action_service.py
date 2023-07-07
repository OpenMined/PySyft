# stdlib
import importlib
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
import numpy as np
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..code.user_code import UserCode
from ..code.user_code import execute_byte_code
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import UserLibConfigRegistry
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .action_object import Action
from .action_object import ActionObject
from .action_object import ActionObjectPointer
from .action_object import ActionType
from .action_object import AnyActionObject
from .action_object import TwinMode
from .action_permissions import ActionObjectREAD
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
            syft_action_data=data, dtype=data.dtype, shape=data.shape
        )
        np_pointer = self.set(context, np_obj)
        return np_pointer

    @service_method(path="action.set", name="set", roles=GUEST_ROLE_LEVEL)
    def set(
        self,
        context: AuthedServiceContext,
        action_object: Union[ActionObject, TwinObject],
    ) -> Result[ActionObject, str]:
        """Save an object to the action store"""
        # 🟡 TODO 9: Create some kind of type checking / protocol for SyftSerializable
        result = self.store.set(
            uid=action_object.id,
            credentials=context.credentials,
            syft_object=action_object,
        )
        if result.is_ok():
            if isinstance(action_object, TwinObject):
                action_object = action_object.mock
            action_object.syft_point_to(context.node.id)
            return Ok(action_object)
        return result.err()

    @service_method(path="action.save", name="save")
    def save(
        self,
        context: AuthedServiceContext,
        action_object: Union[ActionObject, TwinObject],
    ) -> Result[SyftSuccess, str]:
        """Save an object to the action store"""
        # 🟡 TODO 9: Create some kind of type checking / protocol for SyftSerializable
        result = self.store.set(
            uid=action_object.id,
            credentials=context.credentials,
            syft_object=action_object,
        )
        if result.is_ok():
            return Ok(SyftSuccess(message=f"{type(action_object)} saved"))
        return result.err()

    @service_method(path="action.get", name="get", roles=GUEST_ROLE_LEVEL)
    def get(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
    ) -> Result[Ok[ActionObject], Err[str]]:
        """Get an object from the action store"""
        return self._get(context, uid, twin_mode)

    def _get(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
        has_permission=False,
    ) -> Result[ActionObject, str]:
        """Get an object from the action store"""
        result = self.store.get(
            uid=uid, credentials=context.credentials, has_permission=has_permission
        )
        if result.is_ok():
            obj = result.ok()
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
            return Ok(result.ok())
        return Err(result.err())

    # not a public service endpoint
    def _user_code_execute(
        self,
        context: AuthedServiceContext,
        code_item: UserCode,
        kwargs: Dict[str, Any],
    ) -> Result[ActionObjectPointer, Err]:
        filtered_kwargs = code_item.input_policy.filter_kwargs(
            kwargs=kwargs, context=context, code_item_id=code_item.id
        )

        if filtered_kwargs.is_err():
            return filtered_kwargs
        filtered_kwargs = filtered_kwargs.ok()
        has_twin_inputs = False

        real_kwargs = {}
        for key, kwarg_value in filtered_kwargs.items():
            if isinstance(kwarg_value, TwinObject):
                has_twin_inputs = True
            real_kwargs[key] = kwarg_value

        result_id = UID()

        try:
            if not has_twin_inputs:
                # no twins
                filtered_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.NONE
                )
                exec_result = execute_byte_code(code_item, filtered_kwargs)
                result_action_object = wrap_result(result_id, exec_result.result)
            else:
                # twins
                private_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.PRIVATE
                )
                private_exec_result = execute_byte_code(code_item, private_kwargs)
                result_action_object_private = wrap_result(
                    result_id, private_exec_result.result
                )

                mock_kwargs = filter_twin_kwargs(real_kwargs, twin_mode=TwinMode.MOCK)
                mock_exec_result = execute_byte_code(code_item, mock_kwargs)
                result_action_object_mock = wrap_result(
                    result_id, mock_exec_result.result
                )

                result_action_object = TwinObject(
                    id=result_id,
                    private_obj=result_action_object_private,
                    mock_obj=result_action_object_mock,
                )
        except Exception as e:
            return Err(f"_user_code_execute failed. {e}")

        set_result = self.store.set(
            uid=result_id,
            credentials=context.credentials,
            syft_object=result_action_object,
            has_result_read_permission=True,
        )
        if set_result.is_err():
            return set_result.err()
        return Ok(result_action_object)

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
            if action_res.is_err():
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
            setattr(resolved_self.mock, name, public_val)
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
            if action.op == "__call__" and isinstance(
                resolved_self.syft_action_data, Plan
            ):
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

        set_result = self.store.set(
            uid=action.result_id,
            credentials=context.credentials,
            syft_object=result_action_object,
            has_result_read_permission=has_result_read_permission,
        )
        if set_result.is_err():
            return Err(
                f"Failed executing action {action}, set result is an error: {set_result.err()}"
            )

        if isinstance(result_action_object, TwinObject):
            result_action_object = result_action_object.mock
            # we patch this on the object, because this is the thing we are getting back
            result_action_object.id = action.result_id
        result_action_object.syft_point_to(context.node.id)

        return Ok(result_action_object)

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
    args, has_arg_twins = resolve_action_args(action, context, service)
    kwargs, has_kwargs_twins = resolve_action_kwargs(action, context, service)
    if args.is_err():
        return args
    else:
        args = args.ok()
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
                private_args = filter_twin_args(args, twin_mode=twin_mode)
                private_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                private_result = target_method(*private_args, **private_kwargs)
                result_action_object_private = wrap_result(
                    action.result_id, private_result
                )

                mock_args = filter_twin_args(args, twin_mode=twin_mode)
                mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
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
    result_action_object = action_type(id=result_id, syft_action_data=result)
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
