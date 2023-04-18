# stdlib
from enum import Enum
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
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .action_object import Action
from .action_object import ActionObject
from .action_object import ActionObjectPointer
from .action_object import AnyActionObject
from .action_store import ActionStore
from .action_types import action_type_for_type
from .numpy import NumpyArrayObject
from .pandas import PandasDataFrameObject  # noqa: F401
from .pandas import PandasSeriesObject  # noqa: F401


@serializable()
class TwinMode(Enum):
    NONE = 0
    PRIVATE = 1
    MOCK = 2


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
    ) -> Result[ActionObject, str]:
        """Get an object from the action store"""
        # TODO 🟣 Temporarily added skip permission arguments for enclave
        # until permissions are fully integrated
        result = self.store.get(uid=uid, credentials=context.credentials)
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
        return Err(result.err())

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
                result_action_object = wrap_result(
                    code_item.id, result_id, exec_result.result
                )
            else:
                # twins
                private_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.PRIVATE
                )
                private_exec_result = execute_byte_code(code_item, private_kwargs)
                result_action_object_private = wrap_result(
                    code_item.id, result_id, private_exec_result.result
                )

                mock_kwargs = filter_twin_kwargs(real_kwargs, twin_mode=TwinMode.MOCK)
                mock_exec_result = execute_byte_code(code_item, mock_kwargs)
                result_action_object_mock = wrap_result(
                    code_item.id, result_id, mock_exec_result.result
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
        )
        if set_result.is_err():
            return set_result.err()
        return Ok(result_action_object)

    @service_method(path="action.execute", name="execute", roles=GUEST_ROLE_LEVEL)
    def execute(
        self, context: AuthedServiceContext, action: Action
    ) -> Result[ActionObjectPointer, Err]:
        """Execute an operation on objects in the action store"""
        resolved_self = self.get(
            context=context, uid=action.remote_self, twin_mode=TwinMode.NONE
        )
        if resolved_self.is_err():
            return resolved_self.err()
        resolved_self = resolved_self.ok()

        if isinstance(resolved_self, TwinObject):
            private_result = execute_object(
                self, context, resolved_self.private, action, twin_mode=TwinMode.PRIVATE
            )
            if private_result.is_err():
                return private_result.err()
            mock_result = execute_object(
                self, context, resolved_self.mock, action, twin_mode=TwinMode.MOCK
            )
            if mock_result.is_err():
                return mock_result.err()

            private_result = private_result.ok()
            mock_result = mock_result.ok()

            result_action_object = Ok(
                TwinObject(
                    id=action.result_id,
                    private_obj=private_result,
                    private_obj_id=action.result_id,
                    mock_obj=mock_result,
                    mock_obj_id=action.result_id,
                )
            )
        else:
            result_action_object = execute_object(self, context, resolved_self, action)

        if result_action_object.is_err():
            return result_action_object.err()
        else:
            result_action_object = result_action_object.ok()

        set_result = self.store.set(
            uid=action.result_id,
            credentials=context.credentials,
            syft_object=result_action_object,
        )
        if set_result.is_err():
            return set_result.err()

        if isinstance(result_action_object, TwinObject):
            result_action_object = result_action_object.mock
        result_action_object.syft_point_to(context.node.id)

        return Ok(result_action_object)

    @service_method(path="action.exists", name="exists", roles=GUEST_ROLE_LEVEL)
    def exists(
        self, context: AuthedServiceContext, obj_id: UID
    ) -> Result[SyftSuccess, SyftError]:
        """Checks if the given object id exists in the Action Store"""
        if self.store.exists(obj_id):
            return SyftSuccess(message=f"Object: {obj_id} exists")
        else:
            return SyftError(message=f"Object: {obj_id} does not exist")


def execute_object(
    service: ActionService,
    context: AuthedServiceContext,
    resolved_self: ActionObject,
    action: Action,
    twin_mode: TwinMode = TwinMode.NONE,
) -> Result[Union[TwinObject, ActionObject], str]:
    unboxed_resolved_self = resolved_self.syft_action_data
    args = []
    has_twin_inputs = False
    if action.args:
        for arg_id in action.args:
            arg_value = service.get(
                context=context, uid=arg_id, twin_mode=TwinMode.NONE
            )
            if arg_value.is_err():
                return arg_value.err()
            if isinstance(arg_value.ok(), TwinObject):
                has_twin_inputs = True
            args.append(arg_value.ok())

    kwargs = {}
    if action.kwargs:
        for key, arg_id in action.kwargs.items():
            kwarg_value = service.get(
                context=context, uid=arg_id, twin_mode=TwinMode.NONE
            )
            if kwarg_value.is_err():
                return kwarg_value.err()
            if isinstance(kwarg_value.ok(), TwinObject):
                has_twin_inputs = True
            kwargs[key] = kwarg_value.ok()

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
                result_action_object = wrap_result(
                    action.parent_id, action.result_id, result
                )
            elif twin_mode == TwinMode.NONE and has_twin_inputs:
                # self isn't a twin but one of the inputs is
                private_args = filter_twin_args(args, twin_mode=twin_mode)
                private_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                private_result = target_method(*private_args, **private_kwargs)
                result_action_object_private = wrap_result(
                    action.parent_id, action.result_id, private_result
                )

                mock_args = filter_twin_args(args, twin_mode=twin_mode)
                mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                mock_result = target_method(*mock_args, **mock_kwargs)
                result_action_object_mock = wrap_result(
                    action.parent_id, action.result_id, mock_result
                )

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
                result_action_object = wrap_result(
                    action.parent_id, action.result_id, result
                )
            elif twin_mode == twin_mode.MOCK:  # type: ignore
                # twin mock path
                mock_args = filter_twin_args(args, twin_mode=twin_mode)
                mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode)
                target_method = getattr(unboxed_resolved_self, action.op, None)
                result = target_method(*mock_args, **mock_kwargs)
                result_action_object = wrap_result(
                    action.parent_id, action.result_id, result
                )
            else:
                raise Exception(
                    f"Bad combination of: twin_mode: {twin_mode} and has_twin_inputs: {has_twin_inputs}"
                )

    except Exception as e:
        print("what is this exception", e)
        return Err(e)
    return Ok(result_action_object)


def wrap_result(parent_id: UID, result_id: UID, result: Any) -> ActionObject:
    # 🟡 TODO 11: Figure out how we want to store action object results
    action_type = action_type_for_type(result)
    result_action_object = action_type(
        id=result_id, parent_id=parent_id, syft_action_data=result
    )
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
