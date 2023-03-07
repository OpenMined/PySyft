# future
from __future__ import annotations

# stdlib
import inspect
import types
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

# third party
import pydantic
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftBaseObject
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .action_types import action_type_for_type
from .action_types import action_types
from .client import SyftClient
from .response import SyftException


@serializable(recursive_serde=True)
class Action(SyftObject):
    __canonical_name__ = "Action"
    __version__ = SYFT_OBJECT_VERSION_1

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

        action = _self.syft_make_method_action(op=op, args=[other])
        action_result = _self.syft_execute_action(action, sync=True)
        return action_result

    infix_op.__name__ = op
    return infix_op


class ActionObjectPointer:
    pass


# class ActionObjectPointer(SyftObject):
#     __canonical_name__ = "ActionObjectPointer"
#     __version__ = SYFT_OBJECT_VERSION_1

#     __attr_state__ = ["id", "node_uid", "parent_id"]

#     _inflix_operations: List = []

#     node_uid: Optional[UID]
#     parent_id: Optional[UID]

#     def __new__(cls, *args, **kwargs):
#         for op in cls._inflix_operations:
#             new_op = __make_infix_op__(op)
#             setattr(ActionObjectPointer, op, new_op)
#         return super(ActionObjectPointer, cls).__new__(cls)

HOOK_ALWAYS = "ALWAYS"

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
    "send",  # syft
    "_copy_and_set_values",  # pydantic
    "get_from",  # syft
]
dont_wrap_output_attrs = [
    "__repr__",
    "_repr_html_",
    "_repr_markdown_",
    "_repr_latex_",
    "__array_struct__",
    "__array_prepare__",
    "__bool__",
    "__len__",
]
dont_make_side_effects = [
    "_repr_html_",
    "_repr_markdown_",
    "_repr_latex_",
    "__repr__",
    "__getitem__",
    "__setitem__",
    "__len__",
    "shape",
]
show_print = False


def debug_original_func(name, func):
    print(f"{name} func is:")
    print("inspect.isdatadescriptor", inspect.isdatadescriptor(func))
    print("inspect.isgetsetdescriptor", inspect.isgetsetdescriptor(func))
    print("inspect.isfunction", inspect.isfunction(func))
    print("inspect.isbuiltin", inspect.isbuiltin(func))
    print("inspect.ismethod", inspect.ismethod(func))
    print("inspect.ismethoddescriptor", inspect.ismethoddescriptor(func))


def send_action_side_effect(context: PreHookContext, *args: Any, **kwargs: Any) -> Any:
    try:
        if context.op_name not in dont_make_side_effects and hasattr(
            context.obj, "syft_node_uid"
        ):
            if getattr(context.obj, "syft_node_uid", None):
                action = context.obj.syft_make_method_action(
                    op=context.op_name, args=args, kwargs=kwargs
                )
                action_result = context.obj.syft_execute_action(action, sync=True)
                if not isinstance(action_result, ActionObject):
                    print("Got back unexpected response", action_result)
                else:
                    context.node_uid = action_result.syft_node_uid
                    context.result_id = action.result_id
                    print("IGNORING: got action result", action_result)
            else:
                # 🟡 TODO
                pass
                # print(
                #     "Can't Send Action without a target node. Use .point_to(node_uid: UID)"
                # )
    except Exception as e:
        print("Exception in send_action_side_effect", e)
    return context, args, kwargs


def propagate_node_uid(context: PreHookContext, op: str, result: Any) -> Any:
    try:
        if context.op_name not in dont_make_side_effects and hasattr(
            context.obj, "syft_node_uid"
        ):
            syft_node_uid = getattr(context.obj, "syft_node_uid", None)
            if syft_node_uid:
                if not hasattr(result, "syft_node_uid"):
                    print("result doesnt have a syft_node_uid attr")
                if op not in context.obj._syft_dont_wrap_attrs():
                    if hasattr(result, "syft_node_uid"):
                        setattr(result, "syft_node_uid", syft_node_uid)
                else:
                    print("dont propogate node_uid because output isnt wrapped")
            else:
                # 🟡 TODO
                # print("Can't proagate node_uid because parent doesnt have one")
                pass
    except Exception as e:
        print("Exception in propagate_node_uid", e)
    return result


class PreHookContext(SyftBaseObject):
    obj: Any
    op_name: str
    node_uid: Optional[UID]
    result_id: Optional[UID]


class ActionObject(SyftObject):
    __attr_searchable__: List[str] = []
    __canonical_name__ = "ActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_parent_id: Optional[UID]
    syft_pointer_type: ClassVar[Type[ActionObjectPointer]]

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    @pydantic.validator("syft_action_data", pre=True, always=True)
    def check_action_data(
        cls, v: ActionObject.syft_pointer_type
    ) -> ActionObject.syft_pointer_type:
        if cls == AnyActionObject or isinstance(v, cls.syft_internal_type):
            return v
        raise SyftException(
            f"Must init {cls} with {cls.syft_internal_type} not {type(v)}"
        )

    def syft_point_to(self, node_uid: UID) -> None:
        self.syft_node_uid = node_uid

    def syft_get_property(self, obj, method) -> Any:
        klass_method = getattr(type(obj), method, None)
        return klass_method.__get__(obj)

    def syft_is_property(self, obj, method) -> bool:
        klass_method = getattr(type(obj), method, None)
        return isinstance(klass_method, property) or inspect.isdatadescriptor(
            klass_method
        )

    def send(self, client: SyftClient) -> Self:
        return client.api.services.action.set(self)

    def get_from(self, domain_client) -> Any:
        return domain_client.api.services.action.get(self.id).syft_action_data

    @staticmethod
    def from_obj(obj: Any) -> ActionObject:
        action_type = action_type_for_type(obj)
        if action_type is None:
            raise Exception(f"{type(obj)} not in action_types")
        return action_type(syft_action_data=obj)

    syft_action_data: Union[Any, Tuple[Any, Any]]
    syft_node_uid: Optional[UID]
    #  = (
    #     PrivateAttr()
    # )  # 🔵 TODO 6: Make special ActionObject attrs _syft if possible
    _syft_pre_hooks__: Dict[str, Set] = {}
    _syft_post_hooks__: Dict[str, Set] = {}

    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self) -> None:
        if HOOK_ALWAYS not in self._syft_pre_hooks__:
            self._syft_pre_hooks__[HOOK_ALWAYS] = set()
        self._syft_pre_hooks__[HOOK_ALWAYS].add(send_action_side_effect)

        if HOOK_ALWAYS not in self._syft_post_hooks__:
            self._syft_post_hooks__[HOOK_ALWAYS] = set()
        self._syft_post_hooks__[HOOK_ALWAYS].add(propagate_node_uid)

    def _syft_run_pre_hooks__(
        self, context, name, args, kwargs
    ) -> Tuple[PreHookContext, Tuple[Any, ...], Dict[str, Any]]:
        try:
            result_args, result_kwargs = args, kwargs
            if name in self._syft_pre_hooks__:
                for hook in self._syft_pre_hooks__[name]:
                    context, result_args, result_kwargs = hook(
                        context, *result_args, **result_kwargs
                    )

            if name not in self._syft_dont_wrap_attrs():
                if HOOK_ALWAYS in self._syft_pre_hooks__:
                    for hook in self._syft_pre_hooks__[HOOK_ALWAYS]:
                        context, result_args, result_kwargs = hook(
                            context, *result_args, **result_kwargs
                        )
        except Exception as e:
            print("Exception in pre hooks", e)
        return context, result_args, result_kwargs

    def _syft_run_post_hooks__(self, context, name, result):
        new_result = result
        if name in self._syft_post_hooks__:
            for hook in self._syft_post_hooks__[name]:
                new_result = hook(context, name, new_result)

        if name not in self._syft_dont_wrap_attrs():
            if HOOK_ALWAYS in self._syft_post_hooks__:
                for hook in self._syft_post_hooks__[HOOK_ALWAYS]:
                    new_result = hook(context, name, new_result)
        return new_result

    def _syft_output_action_object(self, result) -> Any:
        # can check types here
        if not issubclass(type(result), ActionObject):
            constructor = action_type_for_type(result)
            if not constructor:
                raise Exception(f"output: {type(result)} no in action_types")
            result = constructor(syft_action_data=result)
        return result

    def _syft_passthrough_attrs(self) -> List[str]:
        return passthrough_attrs + getattr(self, "syft_passthrough_attrs", [])

    def _syft_dont_wrap_attrs(self) -> List[str]:
        return dont_wrap_output_attrs + getattr(self, "syft_dont_wrap_attrs", [])

    def __getattribute__(self, name: str) -> Any:
        # bypass certain attrs to prevent recursion issues
        if name.startswith("_syft") or name.startswith("syft"):
            return object.__getattribute__(self, name)

        if name in self._syft_passthrough_attrs():
            return object.__getattribute__(self, name)
        defined_on_self = name in self.__dict__ or name in self.__private_attributes__
        if show_print:
            print(">> ", name, ", defined_on_self = ", defined_on_self)

        # use the custom definied version
        context_self = self
        if not defined_on_self:
            context_self = self.syft_action_data

        # TODO: weird edge cases here for things like tuples
        if name == "__bool__" and not hasattr(self.syft_action_data, "__bool__"):
            context = PreHookContext(obj=self, op_name=name)
            context, _, _ = self._syft_run_pre_hooks__(context, name, (), {})
            # no input needs to propagate
            result = self._syft_run_post_hooks__(
                context, name, bool(self.syft_action_data)
            )
            if name not in self._syft_dont_wrap_attrs():
                result = self._syft_output_action_object(result)

            def __wrapper__bool__() -> bool:
                return result

            return __wrapper__bool__

        # check for other types that aren't methods, functions etc
        if self.syft_is_property(context_self, name):
            context = PreHookContext(obj=self, op_name=name)
            context, _, _ = self._syft_run_pre_hooks__(context, name, (), {})
            # no input needs to propagate
            result = self._syft_run_post_hooks__(
                context, name, self.syft_get_property(context_self, name)
            )
            if name not in self._syft_dont_wrap_attrs():
                result = self._syft_output_action_object(result)
            return result

        original_func = getattr(self.syft_action_data, name)
        if show_print:
            debug_original_func(name, original_func)
        if inspect.ismethod(original_func) or inspect.ismethoddescriptor(original_func):
            if show_print:
                print(">>", name, ", wrapper is method")

            def wrapper(_self, *args, **kwargs):
                context = PreHookContext(obj=self, op_name=name)
                context, pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    context, name, args, kwargs
                )
                result = original_func(*pre_hook_args, **pre_hook_kwargs)
                post_result = self._syft_run_post_hooks__(context, name, result)
                if name not in self._syft_dont_wrap_attrs():
                    post_result = self._syft_output_action_object(post_result)
                    post_result.syft_node_uid = context.node_uid
                    post_result.id = context.result_id
                return post_result

            wrapper = types.MethodType(wrapper, type(self))
        else:
            if show_print:
                print(">>", name, ", wrapper is not method")

            def wrapper(*args, **kwargs):
                context = PreHookContext(obj=self, op_name=name)
                context, pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    context, name, args, kwargs
                )
                result = original_func(*pre_hook_args, **pre_hook_kwargs)
                post_result = self._syft_run_post_hooks__(context, name, result)
                if name not in self._syft_dont_wrap_attrs():
                    post_result = self._syft_output_action_object(post_result)
                    post_result.syft_node_uid = context.node_uid
                    post_result.id = context.result_id
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

    def syft_execute_action(
        self, action: Action, sync: bool = True
    ) -> ActionObjectPointer:
        if self.syft_node_uid is None:
            raise SyftException("Pointers can't execute without a node_uid.")
        # relative
        from .api import APIRegistry
        from .api import SyftAPICall

        api = APIRegistry.api_for(node_uid=self.syft_node_uid)

        kwargs = {"action": action}
        api_call = SyftAPICall(
            node_uid=self.syft_node_uid, path="action.execute", args=[], kwargs=kwargs
        )
        return api.make_call(api_call)

    def syft_make_action(
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

    def syft_make_method_action(
        self,
        op: str,
        args: Optional[List[Union[UID, ActionObjectPointer]]] = None,
        kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]] = None,
    ) -> Action:
        path = self.syft_get_path()
        return self.syft_make_action(
            path=path, op=op, remote_self=self.id, args=args, kwargs=kwargs
        )

    def syft_get_path(self) -> str:
        return f"{type(self).__name__}"

    def syft_remote_method(
        self,
        op: str,
    ) -> Action:
        def wrapper(
            *args: Optional[List[Union[UID, ActionObjectPointer]]],
            **kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]],
        ) -> Action:
            return self.syft_make_method_action(op=op, args=args, kwargs=kwargs)

        return wrapper

    # overwrite the SyftObject implementation
    def keys(self) -> KeysView[str]:
        return self.syft_action_data.keys()

    ###### __DUNDER_MIFFLIN__

    # if we do not implement __add__ then x + y won't trigger __getattribute__
    # no implementation necessary here as we will defer to __getattribute__
    def __repr__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return self.__len__()

    def __getitem__(self, key: Any) -> Any:
        return self._syft_output_action_object(self.__getitem__(key))

    def __eq__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__eq__(other))

    def __setitem__(self, key: Any, value: Any) -> Any:
        return self.__setitem__(key, value)

    def __contains__(self, key: Any) -> bool:
        return self.__contains__(key)

    def __delattr__(self, key: Any) -> None:
        self.__delattr__(key)

    def __delitem__(self, key: Any) -> None:
        self.__delitem__(key)

    def __invert__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__invert__(other))

    def __divmod__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__add__(other))

    def __floordiv__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__floordiv__(other))

    def __mod__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__mod__(other))

    def __bool__(self) -> bool:
        return self.__bool__()

    def __add__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__add__(other))

    def __mul__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__mul__(other))

    def __matmul__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__matmul__(other))


@serializable(recursive_serde=True)
class AnyActionObject(ActionObject):
    __canonical_name__ = "AnyActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: Any = Any
    syft_passthrough_attrs = []
    syft_dont_wrap_attrs = []

    def __float__(self) -> float:
        return float(self.syft_action_data)


action_types[Any] = AnyActionObject
