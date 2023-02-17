# stdlib
from typing import Any

# third party
import numpy as np
from result import Err
from result import Ok
from result import Result

# relative
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .action_object import Action
from .action_object import ActionObject
from .action_object import ActionObjectPointer
from .action_store import ActionStore
from .context import AuthedServiceContext
from .numpy_array import NumpyArrayObject
from .service import AbstractService
from .service import service_method


@serializable(recursive_serde=True)
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

    @service_method(path="action.set", name="set")
    def set(
        self, context: AuthedServiceContext, action_object: ActionObject
    ) -> Result[ActionObjectPointer, str]:
        """Save an object to the action store"""
        # 🟡 TODO 9: Create some kind of type checking / protocol for SyftSerializable
        result = self.store.set(
            uid=action_object.id,
            credentials=context.credentials,
            syft_object=action_object,
        )
        if result.is_ok():
            return Ok(action_object.to_pointer(context.node.id))
        return result.err()

    @service_method(path="action.get", name="get")
    def get(self, context: AuthedServiceContext, uid: UID) -> Result[ActionObject, str]:
        """Get an object from the action store"""
        result = self.store.get(uid=uid, credentials=context.credentials)
        if result.is_ok():
            return Ok(result.ok())
        return Err(result.err())

    @service_method(path="action.get_pointer", name="get_pointer")
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

    @service_method(path="action.execute", name="execute")
    def execute(
        self, context: AuthedServiceContext, action: Action
    ) -> Result[ActionObjectPointer, Err]:
        """Execute an operation on objects in the action store"""
        resolved_self = self.get(context=context, uid=action.remote_self)
        if resolved_self.is_err():
            return resolved_self.err()
        else:
            resolved_self = resolved_self.ok().syft_action_data
        args = []
        if action.args:
            for arg_id in action.args:
                arg_value = self.get(context=context, uid=arg_id)
                if arg_value.is_err():
                    return arg_value.err()
                args.append(arg_value.ok().syft_action_data)

        kwargs = {}
        if action.kwargs:
            for key, arg_id in action.kwargs.items():
                kwarg_value = self.get(context=context, uid=arg_id)
                if kwarg_value.is_err():
                    return kwarg_value.err()
                kwargs[key] = kwarg_value.ok().syft_action_data

        # 🔵 TODO 10: Get proper code From old RunClassMethodAction to ensure the function
        # is not bound to the original object or mutated
        target_method = getattr(resolved_self, action.op, None)
        result = None
        try:
            if target_method:
                result = target_method(*args, **kwargs)
        except Exception as e:
            return Err(e)

        # 🟡 TODO 11: Figure out how we want to store action object results
        if isinstance(result, np.ndarray):
            result_action_object = NumpyArrayObject(
                id=action.result_id, parent_id=action.id, syft_action_data=result
            )
        else:
            # 🔵 TODO 12: Create an AnyPointer to handle unexpected results
            result_action_object = ActionObject(
                id=action.result_id, parent_id=action.id, syft_action_data=result  # type: ignore
            )

        set_result = self.store.set(
            uid=action.result_id,
            credentials=context.credentials,
            syft_object=result_action_object,
        )
        if set_result.is_err():
            return set_result.err()

        return Ok(result_action_object.to_pointer(node_uid=context.node.id))
