# stdlib
import types
from typing import Any
from typing import Callable
from typing import List
from typing import Optional

# third party
import numpy as np
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import transform
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .action_object import Action
from .action_object import ActionObject
from .action_object import ActionObjectPointer
from .action_store import ActionStore
from .credentials import SyftVerifyKey


@serializable(recursive_serde=True)
class NumpyArrayObjectPointer(ActionObjectPointer):
    __canonical_name__ = "NumpyArrayObjectPointer"
    __version__ = 1

    public_dtype: Optional[str]  # 🟡 TODO: add numpy dtype types to serde
    public_shape: Optional[tuple]

    # 🟡 TODO: add state / allowlist inheritance and ignore methods by default
    __attr_state__ = ["id", "node_uid", "parent_id", "public_dtype", "public_shape"]

    def __post_init__(self) -> None:
        self.setup_methods()

    def setup_methods(self) -> None:
        infix_operations = ["__add__", "__sub__"]
        for op in infix_operations:
            setattr(
                type(self),
                op,
                types.MethodType(self.__make_infix_op__(op), type(self)),
            )

    def __make_infix_op__(self, op: str) -> Callable:
        def infix_op(_self, other: ActionObjectPointer) -> Self:
            if not isinstance(other, ActionObjectPointer):
                print("TODO: pointerize")
                raise Exception("We need to pointerize first")
            action = self.make_method_action(op=op, args=[other])
            action_result = self.execute_action(action, sync=True)
            return action_result

        infix_op.__name__ = op
        return infix_op


def numpy_like_eq(left: Any, right: Any) -> bool:
    result = left == right
    if isinstance(result, bool):
        return result

    if hasattr(result, "all"):
        return (result).all()
    return bool(result)


# 🟡 TODO: Map numpy type to these classes for bi-directional lookup
@serializable(recursive_serde=True)
class NumpyArrayObject(ActionObject):
    __canonical_name__ = "NumpyArrayObject"
    __version__ = 1

    dtype: str  # 🟡 TODO: add numpy dtype types to serde
    shape: tuple

    pointer_type = NumpyArrayObjectPointer

    def __eq__(self, other: Any) -> bool:
        # 🟡 TODO: move to a Data / Serdeable type interface on ActionObject
        if isinstance(other, NumpyArrayObject):
            return (
                numpy_like_eq(self.data, other.data)
                and self.dtype == other.dtype
                and self.shape == other.shape
                and self.pointer_type == other.pointer_type
            )
        return self == other


def expose_dtype(output: dict) -> dict:
    output["public_dtype"] = output["dtype"]
    del output["dtype"]
    return output


def expose_shape(output: dict) -> dict:
    output["public_shape"] = output["shape"]
    del output["shape"]
    return output


@transform(NumpyArrayObject, NumpyArrayObjectPointer)
def np_array_to_pointer() -> List[Callable]:
    return [
        expose_dtype,
        expose_shape,
    ]


class ActionService:
    def __init__(self, node_uid: UID, store: ActionStore = ActionStore()) -> None:
        self.node_uid = node_uid
        self.store = store

    # @service(path="services.happy.maybe_create", name="create_user")
    def set(
        self, credentials: SyftVerifyKey, action_object: ActionObject
    ) -> Result[ActionObjectPointer, str]:
        """Save an object to the action store"""

        # 🟡 TODO: Create some kind of type checking / protocol for is_serializable???
        result = self.store.set(
            uid=action_object.id,
            credentials=credentials,
            syft_object=action_object,
        )
        if result.is_ok():
            return Ok(action_object.to_pointer(self.node_uid))
        return result.err()

    def get(self, credentials: SyftVerifyKey, uid: UID) -> Result[ActionObject, str]:
        """Get an object from the action store"""
        result = self.store.get(uid=uid, credentials=credentials)
        if result.is_ok():
            return Ok(result.ok())
        return Err(result.err())

    def execute(
        self, credentials: SyftVerifyKey, action: Action
    ) -> Result[ActionObjectPointer, Err]:
        """Execute an operation on objects in the action store"""
        resolved_self = self.get(credentials=credentials, uid=action.remote_self)
        if resolved_self.is_err():
            return resolved_self.err()
        else:
            resolved_self = resolved_self.ok().data
        args = []
        if action.args:
            for arg_id in action.args:
                arg_value = self.get(credentials=credentials, uid=arg_id)
                if arg_value.is_err():
                    return arg_value.err()
                args.append(arg_value.ok().data)

        kwargs = {}
        if action.kwargs:
            for key, arg_id in action.kwargs.items():
                kwarg_value = self.get(credentials=credentials, uid=arg_id)
                if kwarg_value.is_err():
                    return kwarg_value.err()
                kwargs[key] = kwarg_value.ok().data

        # 🟡 TODO: GET PROPER CODE FROM OLD RUNCLASSMETHODACTION to ensure the function
        # is not bound to the original object or mutated
        target_method = getattr(resolved_self, action.op, None)
        result = None
        try:
            if target_method:
                result = target_method(*args, **kwargs)
        except Exception as e:
            print("what is this exception", e)
            return Err(e)

        # 🟡 TODO: Figure out how we want to store action object results
        # add a mapping between types and their synthesized ActionObjects

        if isinstance(result, np.ndarray):
            result_action_object = NumpyArrayObject(
                id=action.result_id,
                parent_id=action.id,
                data=result,
                dtype=str(result.dtype),
                shape=result.shape,
            )
        else:
            # 🟡 TODO: we need an Any Pointer?
            result_action_object = ActionObject(
                id=action.result_id, parent_id=action.id, data=result
            )

        set_result = self.store.set(
            uid=action.result_id,
            credentials=credentials,
            syft_object=result_action_object,
        )
        if set_result.is_err():
            return set_result.err()

        return Ok(result_action_object.to_pointer(self.node_uid))
