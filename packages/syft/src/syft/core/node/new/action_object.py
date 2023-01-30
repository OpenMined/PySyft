# future
from __future__ import annotations

# stdlib
import inspect
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
import pydantic
from typing_extensions import Self

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


def __make_infix_op__(op: str) -> Callable:
    def infix_op(_self, other: Any) -> Self:
        if not isinstance(other, ActionObjectPointer):
            other = other.to_pointer(_self.node_uid)

        action = _self.make_method_action(op=op, args=[other])
        action_result = _self.execute_action(action, sync=True)
        return action_result

    infix_op.__name__ = op
    return infix_op


class ActionObjectPointer(SyftObject):
    __canonical_name__ = "ActionObjectPointer"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_state__ = ["id", "node_uid", "parent_id"]

    _inflix_operations: List = []

    node_uid: Optional[UID]
    parent_id: Optional[UID]

    def __new__(cls, *args, **kwargs):
        for op in cls._inflix_operations:
            new_op = __make_infix_op__(op)
            setattr(ActionObjectPointer, op, new_op)
        return super(ActionObjectPointer, cls).__new__(cls)

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


class ActionObject(SyftObject):
    __attr_searchable__: List[str] = []
    __canonical_name__ = "ActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_parent_id: Optional[UID]
    syft_pointer_type: ClassVar[Type[ActionObjectPointer]]

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    def to_pointer(self, node_uid: UID) -> syft_pointer_type:
        pointer = self.to(self.syft_pointer_type)
        pointer.node_uid = node_uid
        return pointer

    syft_action_data: Any
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
        print("custom repr")
        return self.__repr__()

    def __post_init__(self) -> None:
        pass

    def __eq__(self, other: Any) -> bool:
        return self.__add__(other)

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

    def _syft_output_action_object(self, result) -> Any:
        # can check types here
        if not isinstance(result, ActionObject):
            result = ActionObject(syft_action_data=result)
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
            context_self = self.syft_action_data

        if is_property(context_self, name):
            _ = self._syft_run_pre_hooks__(name, (), {})
            # no input needs to propagate
            result = self._syft_run_post_hooks__(
                name, object.__getattribute__(context_self, name)
            )
            if name not in dont_wrap_output_attrs:
                result = self._syft_output_action_object(result)
            return result

        # check for other types that aren't methods, functions etc
        original_func = getattr(self.syft_action_data, name)
        if show_print:
            debug_original_func(name, original_func)
        if inspect.ismethod(original_func) or inspect.ismethoddescriptor(original_func):
            if show_print:
                print(">>", name, ", wrapper is method")

            def wrapper(self, *args, **kwargs):
                pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    name, args, kwargs
                )
                result = original_func(*pre_hook_args, **pre_hook_kwargs)
                post_result = self._syft_run_post_hooks__(name, result)
                if name not in dont_wrap_output_attrs:
                    post_result = self._syft_output_action_object(post_result)
                return post_result

        else:
            if show_print:
                print(">>", name, ", wrapper is not method")

            def wrapper(*args, **kwargs):
                pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    name, args, kwargs
                )
                result = original_func(*pre_hook_args, **pre_hook_kwargs)
                post_result = self._syft_run_post_hooks__(name, result)
                if name not in dont_wrap_output_attrs:
                    post_result = self._syft_output_action_object(post_result)
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
