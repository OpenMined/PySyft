# future
from __future__ import annotations

# stdlib
import inspect
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# third party
import numpy as np
import pydantic

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID


@serializable(recursive_serde=True)
class Action(SyftObject):
    __attr_searchable__: List[str] = []

    parent_id: Optional[UID]

    path: str
    op: str
    remote_self: Optional[UID]
    args: List[UID]
    kwargs: Dict[str, UID]
    result_id: Optional[UID]

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    @pydantic.validator("result_id", pre=True, always=True)
    def make_result_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    @property
    def full_path(self) -> str:
        return f"{self.path}.{self.op}"


class ActionObjectPointer(SyftObject, extra=pydantic.Extra.allow):
    __canonical_name__ = "ActionObjectPointer"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_state__ = ["id", "node_uid", "parent_id"]

    node_uid: Optional[UID]
    parent_id: Optional[UID]

    def execute_action(self, action: Action, sync: bool = True) -> ActionObjectPointer:
        if self.node_uid is None:
            raise Exception("Pointers can't execute without a node_uid.")
        # relative
        from .api import APIRegistry
        from .api import SyftAPICall

        api = APIRegistry.api_for(node_uid=self.node_uid)

        kwargs = {"action": action}
        api_call = SyftAPICall(path="action.execute", args=[], kwargs=kwargs)
        return api.make_call(api_call)

    def make_action(
        self,
        path: str,
        op: str,
        remote_self: Optional[UID] = None,
        args: Optional[List[Union[UID, ActionObjectPointer]]] = None,
        kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]] = None,
    ) -> Action:
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        arg_ids = [uid if isinstance(uid, UID) else uid.id for uid in args]
        kwarg_ids = {
            k: uid if isinstance(uid, UID) else uid.id for k, uid in kwargs.items()
        }
        return Action(
            parent_id=self.id,
            path=path,
            op=op,
            remote_self=remote_self,
            args=arg_ids,
            kwargs=kwarg_ids,
        )

    def make_method_action(
        self,
        op: str,
        args: Optional[List[Union[UID, ActionObjectPointer]]] = None,
        kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]] = None,
    ) -> Action:
        path = self.get_path()
        return self.make_action(
            path=path, op=op, remote_self=self.id, args=args, kwargs=kwargs
        )

    def get_path(self) -> str:
        return f"{type(self).__name__}"

    def remote_method(
        self,
        op: str,
    ) -> Action:
        def wrapper(
            *args: Optional[List[Union[UID, ActionObjectPointer]]],
            **kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]],
        ) -> Action:
            return self.make_method_action(op=op, args=args, kwargs=kwargs)

        return wrapper


passthrough_attrs = [
    "__dict__",  # python
    "__class__",  # python
    "__repr_name__",  # python
    "__annotations__",  # python
    "_init_private_attributes",  # pydantic
    "__private_attributes__",  # pydantic
    "__config__",  # pydantic
    "__fields__",  # pydantic
    "__fields_set__",  # pydantic
    "__repr_str__",  # pydantic
    "__repr_args__",  # pydantic
    "__post_init__",  # syft
    "id",  # syft
    "to_mongo",  # syft 🟡 TODO 23: Add composeable / inheritable object passthrough attrs
    "__attr_searchable__",  # syft
    "__canonical_name__",  # syft
    "__version__",  # syft
    "__args__",  # pydantic
    "to_pointer",  # syft
    "to",  # syft
]
dont_wrap_output_attrs = ["__repr__", "__array_struct__", "__array_prepare__"]

show_print = False


def debug_original_func(name, func):
    print(f"{name} func is:")
    print("inspect.isdatadescriptor", inspect.isdatadescriptor(func))
    print("inspect.isgetsetdescriptor", inspect.isgetsetdescriptor(func))
    print("inspect.isfunction", inspect.isfunction(func))
    print("inspect.isbuiltin", inspect.isbuiltin(func))
    print("inspect.ismethod", inspect.ismethod(func))
    print("inspect.ismethoddescriptor", inspect.ismethoddescriptor(func))


def is_property(obj, method) -> bool:
    klass_method = getattr(type(obj), method, None)
    return inspect.isgetsetdescriptor(klass_method)


def get_property(obj, method) -> Any:
    klass_method = getattr(type(obj), method, None)
    return klass_method.__get__(obj)


def hash_inputs(
    sequence: Union[Dict, List], name: str, ids: List, other: List
) -> Tuple[List, List]:
    """This method iterates through a function's args and kwargs and creates hashes used for a History Hash"""
    if isinstance(sequence, Dict):
        if not sequence:
            return ids, other  # we were asked to hash kwargs but none were provided.
        else:
            sequence = [v for k, v in sequence.items()]

    for item in sequence:
        if isinstance(item, ActionObject):
            ids.append(item.id)
        elif isinstance(item, Hashable):
            other.append(hash(item))
        elif isinstance(item, np.ndarray):
            other.append(hash(item.tobytes()))  # this could be slow for large np arrays
        else:
            raise NotImplementedError(
                f"Unable to hash parent object: {type(item)} in method: {name}"
            )
    return ids, other


def fetch_all_inputs(
    name: str, self_id: UID, args, kwargs
) -> Tuple[Optional[List], Optional[List], Optional[List]]:
    """
    Returns everything needed to create a History Hash for the resultant ActionObject:
    - a List of Parent IDs
    - a List of input arguments
    - a List of input kwargs
    """
    parent_ids = [self_id]
    parent_ids, parent_args = hash_inputs(
        sequence=args, name=name, ids=parent_ids, other=[]
    )
    parent_ids, parent_kwargs = hash_inputs(
        sequence=kwargs, name=name, ids=parent_ids, other=[]
    )
    return parent_ids, parent_args, parent_kwargs


class ActionObject(SyftObject):
    __attr_searchable__: List[str] = []
    __canonical_name__ = "ActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_action_data: Optional[Any] = None
    syft_pointer_type: ClassVar[Type[ActionObjectPointer]]

    # Help with calculating history hash for code verification
    syft_parent_id: Optional[Union[UID, List[UID]]]
    syft_parent_op: Optional[str]
    syft_parent_args: Optional[Any]
    syft_parent_kwargs: Optional[Any]
    syft_history_hash: Optional[int]
    syft_result_obj: Optional[Any]

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    def to_pointer(self, node_uid: UID) -> syft_pointer_type:
        pointer = self.to(self.syft_pointer_type)
        pointer.node_uid = node_uid
        return pointer

    #  = (
    #     PrivateAttr()
    # )  # 🔵 TODO 6: Make special ActionObject attrs _syft if possible
    syft_pre_hooks__: Dict[str, Any] = {}
    syft_post_hooks__: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

    # if we do not implement __add__ then x + y won't trigger __getattribute__
    # no implementation necessary here as we will defer to __getattribute__
    def __add__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__add__(other))

    def __repr__(self) -> str:
        return f"History: {self.syft_history_hash}"

    def __post_init__(self) -> None:
        if self.syft_parent_id is not None:
            if not isinstance(self.syft_parent_id, list):
                raise NotImplementedError(
                    f"Parent ID type not recognized: {type(self.syft_parent_id)}"
                )
            else:
                print("We're making history... hashes.")
                # This Action Object has 1+ parent so it'll need a history hash for verification later
                history = ""
                for parent in self.syft_parent_id:
                    history += str(parent)
                if self.syft_parent_op is not None:
                    history += self.syft_parent_op
                if self.syft_parent_args is not None:
                    if isinstance(self.syft_parent_args, list):
                        for arg in self.syft_parent_args:
                            history += str(arg)
                    else:
                        history += str(self.syft_parent_args)
                if self.syft_parent_kwargs is not None:
                    if isinstance(self.syft_parent_kwargs, list):
                        for kwarg in self.syft_parent_kwargs:
                            history += str(kwarg)
                    else:
                        history += str(self.syft_parent_kwargs)
                # else:
                #     print("We have no Parent args!")
                self.syft_history_hash = hash(history)
        print("Lights... Cameras... ACTION OBJECT!")

    def __eq__(self, other: Any) -> bool:
        return self._syft_output_action_object(self.__eq__(other))

    def _syft_run_pre_hooks__(self, name, args, kwargs):
        result_args, result_kwargs = args, kwargs
        if name in self.syft_pre_hooks__:
            for hook in self.syft_pre_hooks__[name]:
                print(
                    f"Running {name} syft_run_pre_hooks__", result_args, result_kwargs
                )
                result_args, result_kwargs = hook(*args, **kwargs)
            print(f"Returning {name} syft_run_pre_hooks__", result_args, result_kwargs)
        return result_args, result_kwargs

    def _syft_run_post_hooks__(self, name, result):
        new_result = result
        if name in self.syft_post_hooks__:
            for hook in self.syft_post_hooks__[name]:
                print(f"Running {name} syft_post_hooks__", new_result)
                new_result = hook(result)
            print(f"Returning {name} syft_post_hooks__", new_result)
        return new_result

    def _syft_output_action_object(
        self,
        result,
        parents_id: Optional[Union[UID, List[UID]]] = None,
        op_name: Optional[str] = None,
        parent_args: Optional[str, List[str]] = None,
        parent_kwargs: Optional[str, List[str]] = None,
    ) -> Any:
        """Given an input argument (result) this method ensures the output is an ActionObject as well."""
        # can check types here
        if not isinstance(result, ActionObject):
            if parents_id is None:
                parents_id = self.id
            elif isinstance(parents_id, list):
                if self.id not in parents_id:
                    parents_id.append(self.id)
            elif isinstance(parents_id, UID):
                if parents_id != self.id:
                    parents_id = [parents_id, self.id]
            else:
                raise NotImplementedError(
                    "Not implemented for Parent_id type: ", type(parents_id), parents_id
                )

            result = ActionObject(
                syft_action_data=result,
                syft_parent_id=parents_id,
                syft_parent_op=op_name,
                syft_parent_args=parent_args,
                syft_parent_kwargs=parent_kwargs,
            )

        return result

    def __getattribute__(self, name):
        # bypass certain attrs to prevent recursion issues
        if (
            name in passthrough_attrs
            or name.startswith("_syft")
            or name.startswith("syft")
        ):
            return object.__getattribute__(self, name)

        defined_on_self = name in self.__dict__ or name in self.__private_attributes__
        if show_print:
            print(">> ", name, ", defined_on_self = ", defined_on_self)

        # use the custom definied version
        context_self = self
        if not defined_on_self:
            if self.syft_action_data is not None:
                context_self = self.syft_action_data
            else:
                context_self = self.syft_result_obj
                # raise NotImplementedError(f"OP: {name}, id: {self.id}")

        if is_property(context_self, name):
            _ = self._syft_run_pre_hooks__(name, (), {})
            # no input needs to propagate
            result = self._syft_run_post_hooks__(
                name, object.__getattribute__(context_self, name)
            )
            if name not in dont_wrap_output_attrs:
                result = self._syft_output_action_object(
                    result, parents_id=self.id, op_name=name
                )
            return result

        # check for other types that aren't methods, functions etc
        if self.syft_action_data is not None:
            original_func = getattr(self.syft_action_data, name)
            skip_result = False
        else:
            original_func = getattr(self.syft_result_obj, name)
            skip_result = True

        if show_print:
            debug_original_func(name, original_func)
        if inspect.ismethod(original_func) or inspect.ismethoddescriptor(original_func):
            if show_print:
                print(">>", name, ", wrapper is method")

            def wrapper(self, *args, **kwargs):
                pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    name, args, kwargs
                )
                if skip_result:
                    result = None
                else:
                    result = original_func(*pre_hook_args, **pre_hook_kwargs)

                post_result = self._syft_run_post_hooks__(name, result)
                if name not in dont_wrap_output_attrs:
                    parent_ids, parent_args, parent_kwargs = fetch_all_inputs(
                        name, self.id, args, kwargs
                    )
                    post_result = self._syft_output_action_object(
                        post_result,
                        parents_id=parent_ids,
                        op_name=name,
                        parent_args=parent_args,
                        parent_kwargs=parent_kwargs,
                    )
                return post_result

        else:
            if show_print:
                print(">>", name, ", wrapper is not method")

            def wrapper(*args, **kwargs):
                pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    name, args, kwargs
                )
                if skip_result:
                    result = None
                else:
                    result = original_func(*pre_hook_args, **pre_hook_kwargs)

                post_result = self._syft_run_post_hooks__(name, result)
                if name not in dont_wrap_output_attrs:
                    parent_ids, parent_args, parent_kwargs = fetch_all_inputs(
                        name, self.id, args, kwargs
                    )
                    post_result = self._syft_output_action_object(
                        post_result,
                        parents_id=parent_ids,
                        op_name=name,
                        parent_args=parent_args,
                        parent_kwargs=parent_kwargs,
                    )
                return post_result

        try:
            wrapper.__doc__ = original_func.__doc__
            if show_print:
                print(
                    "Found original signature for ",
                    name,
                    inspect.signature(original_func),
                )
            wrapper.__ipython_inspector_signature_override__ = inspect.signature(
                original_func
            )
        except Exception:
            if show_print:
                print("name", name, "has no signature")

        return wrapper
