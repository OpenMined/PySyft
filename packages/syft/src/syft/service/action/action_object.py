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
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject
from ...types.syft_object import SyftObject
from ...types.uid import LineageID
from ...types.uid import UID
from ..response import SyftException
from .action_data_empty import ActionDataEmpty
from .action_types import action_type_for_type
from .action_types import action_types


@serializable()
class Action(SyftObject):
    __canonical_name__ = "Action"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__: List[str] = []

    path: str
    op: str
    remote_self: Optional[LineageID]
    args: List[LineageID]
    kwargs: Dict[str, LineageID]
    result_id: Optional[LineageID]

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    @pydantic.validator("result_id", pre=True, always=True)
    def make_result_id(cls, v: Optional[Union[UID, LineageID]]) -> UID:
        return v if isinstance(v, LineageID) else LineageID(v)

    @property
    def full_path(self) -> str:
        return f"{self.path}.{self.op}"

    @property
    def syft_history_hash(self) -> int:
        hashes = 0
        if self.remote_self:
            hashes += hash(self.remote_self.syft_history_hash)
        # 🔵 TODO: resolve this
        # if the object is ActionDataEmpty then the type might not be equal to the
        # real thing. This is the same issue with determining the result type from
        # a pointer operation in the past, so we should think about what we want here
        # hashes += hash(self.path)
        hashes += hash(self.op)
        for arg in self.args:
            hashes += hash(arg.syft_history_hash)
        for k, arg in self.kwargs.items():
            hashes += hash(k)
            hashes += hash(arg.syft_history_hash)
        return hashes


class ActionObjectPointer:
    pass


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
    "__str__",
    "_repr_html_",
    "_repr_markdown_",
    "_repr_latex_",
    "__array_struct__",
    "__array_prepare__",
    "__array_wrap__",
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
action_data_empty_must_run = [
    "__repr__",
]
show_print = False


def debug_original_func(name: str, func: Callable) -> None:
    print(f"{name} func is:")
    print("inspect.isdatadescriptor", inspect.isdatadescriptor(func))
    print("inspect.isgetsetdescriptor", inspect.isgetsetdescriptor(func))
    print("inspect.isfunction", inspect.isfunction(func))
    print("inspect.isbuiltin", inspect.isbuiltin(func))
    print("inspect.ismethod", inspect.ismethod(func))
    print("inspect.ismethoddescriptor", inspect.ismethoddescriptor(func))


def make_action_side_effect(context: PreHookContext, *args: Any, **kwargs: Any) -> Any:
    try:
        action = context.obj.syft_make_method_action(
            op=context.op_name, args=args, kwargs=kwargs
        )
        context.action = action
    except Exception:  # nosec
        # print(
        #     "Exception detected in make_action_side_effect", e
        # )  # TODO: Put this Exception back
        pass
    return context, args, kwargs


def send_action_side_effect(context: PreHookContext, *args: Any, **kwargs: Any) -> Any:
    try:
        if context.op_name not in dont_make_side_effects and hasattr(
            context.obj, "syft_node_uid"
        ):
            if getattr(context.obj, "syft_node_uid", None):
                if context.action is not None:
                    action = context.obj.syft_make_method_action(
                        op=context.op_name, args=args, kwargs=kwargs
                    )
                    context.action = action

                action_result = context.obj.syft_execute_action(action, sync=True)
                if not isinstance(action_result, ActionObject):
                    # print("Got back unexpected response", action_result)
                    pass
                else:
                    context.node_uid = action_result.syft_node_uid
                    context.result_id = action.result_id
                    # print("IGNORING: got action result", action_result)
            else:
                # 🟡 TODO
                pass
                # print(
                #     "Can't Send Action without a target node. Use .point_to(node_uid: UID)"
                # )
    except Exception:  # nosec
        # print(
        #     "Exception in send_action_side_effect", e
        # )  # TODO: Put this Exception back
        pass
    return context, args, kwargs


def propagate_node_uid(context: PreHookContext, op: str, result: Any) -> Any:
    try:
        if context.op_name not in dont_make_side_effects and hasattr(
            context.obj, "syft_node_uid"
        ):
            syft_node_uid = getattr(context.obj, "syft_node_uid", None)
            if syft_node_uid:
                if not hasattr(result, "syft_node_uid"):
                    # print("result doesnt have a syft_node_uid attr")
                    pass
                if op not in context.obj._syft_dont_wrap_attrs():
                    if hasattr(result, "syft_node_uid"):
                        setattr(result, "syft_node_uid", syft_node_uid)
                else:
                    # print("dont propogate node_uid because output isnt wrapped")
                    pass
            else:
                # 🟡 TODO
                # print("Can't proagate node_uid because parent doesnt have one")
                pass
    except Exception:  # nosec
        # print("Exception in propagate_node_uid", e)
        pass
    return result


def is_action_data_empty(obj: Any) -> bool:
    if hasattr(obj, "syft_action_data"):
        obj = obj.syft_action_data
    if isinstance(obj, ActionDataEmpty):
        return True
    return False


def has_action_data_empty(args: Any, kwargs: Any) -> bool:
    for a in args:
        if is_action_data_empty(a):
            return True

    for _, a in kwargs.items():
        if is_action_data_empty(a):
            return True
    return False


def debox_args_and_kwargs(args: Any, kwargs: Any) -> Tuple[Any, Any]:
    filtered_args = []
    filtered_kwargs = {}
    for a in args:
        value = a
        if hasattr(value, "syft_action_data"):
            value = value.syft_action_data
        filtered_args.append(value)

    for k, a in kwargs.items():
        value = a
        if hasattr(value, "syft_action_data"):
            value = value.syft_action_data
        filtered_kwargs[k] = a

    return tuple(filtered_args), filtered_kwargs


class PreHookContext(SyftBaseObject):
    obj: Any
    op_name: str
    node_uid: Optional[UID]
    result_id: Optional[Union[UID, LineageID]]
    action: Optional[Action]


class ActionObject(SyftObject):
    __canonical_name__ = "ActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__: List[str] = []
    syft_action_data: Optional[Any] = None
    syft_pointer_type: ClassVar[Type[ActionObjectPointer]]

    # Help with calculating history hash for code verification
    syft_parent_hashes: Optional[Union[int, List[int]]]
    syft_parent_op: Optional[str]
    syft_parent_args: Optional[Any]
    syft_parent_kwargs: Optional[Any]
    syft_history_hash: Optional[int]
    syft_internal_type: ClassVar[Type[Any]]
    syft_node_uid: Optional[UID]
    _syft_pre_hooks__: Dict[str, Set] = {}
    _syft_post_hooks__: Dict[str, Set] = {}

    @property
    def syft_lineage_id(self) -> LineageID:
        return LineageID(self.id, self.syft_history_hash)

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    @pydantic.validator("syft_action_data", pre=True, always=True)
    def check_action_data(
        cls, v: ActionObject.syft_pointer_type
    ) -> ActionObject.syft_pointer_type:
        if cls == AnyActionObject or isinstance(
            v, (cls.syft_internal_type, ActionDataEmpty)
        ):
            return v
        raise SyftException(
            f"Must init {cls} with {cls.syft_internal_type} not {type(v)}"
        )

    def syft_point_to(self, node_uid: UID) -> None:
        self.syft_node_uid = node_uid

    def syft_get_property(self, obj: Any, method: str) -> Any:
        klass_method = getattr(type(obj), method, None)
        if klass_method is None:
            raise Exception(f"{type(obj)} has no {method} attribute")
        return klass_method.__get__(obj)

    def syft_is_property(self, obj: Any, method: str) -> bool:
        klass_method = getattr(type(obj), method, None)
        return isinstance(klass_method, property) or inspect.isdatadescriptor(
            klass_method
        )

    def send(self, client: SyftClient) -> Self:
        return client.api.services.action.set(self)

    def get_from(self, client: SyftClient) -> Any:
        return client.api.services.action.get(self.id).syft_action_data

    @staticmethod
    def from_obj(
        syft_action_data: Any,
        id: Optional[UID] = None,
        syft_lineage_id: Optional[LineageID] = None,
    ) -> ActionObject:
        if isinstance(syft_action_data, ActionObject):
            return syft_action_data
        if id and syft_lineage_id and id != syft_lineage_id.id:
            raise Exception("UID and LineageID should match")
        action_type = action_type_for_type(syft_action_data)
        if action_type is None:
            raise Exception(f"{type(syft_action_data)} not in action_types")
        action_object = action_type(syft_action_data=syft_action_data)
        if id:
            action_object.id = id
        if syft_lineage_id:
            action_object.id = syft_lineage_id.id
            action_object.syft_history_hash = syft_lineage_id.syft_history_hash
        elif id:
            action_object.syft_history_hash = hash(id)

        return action_object

    @staticmethod
    def empty(
        syft_internal_type: Optional[Any] = Any,
        id: Optional[UID] = None,
        syft_lineage_id: Optional[LineageID] = None,
    ) -> ActionObject:
        empty = ActionDataEmpty(syft_internal_type=syft_internal_type)
        action_object = ActionObject.from_obj(
            syft_action_data=empty, id=id, syft_lineage_id=syft_lineage_id
        )
        return action_object

    def __post_init__(self) -> None:
        if HOOK_ALWAYS not in self._syft_pre_hooks__:
            self._syft_pre_hooks__[HOOK_ALWAYS] = set()
        self._syft_pre_hooks__[HOOK_ALWAYS].add(make_action_side_effect)
        self._syft_pre_hooks__[HOOK_ALWAYS].add(send_action_side_effect)

        if HOOK_ALWAYS not in self._syft_post_hooks__:
            self._syft_post_hooks__[HOOK_ALWAYS] = set()
        self._syft_post_hooks__[HOOK_ALWAYS].add(propagate_node_uid)

        if isinstance(self.syft_action_data, ActionObject):
            raise Exception("Nested ActionObjects", self.syft_action_data)

        self.syft_history_hash = hash(self.id)

    def _syft_run_pre_hooks__(
        self, context: PreHookContext, name: str, args: Any, kwargs: Any
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

    def _syft_run_post_hooks__(
        self, context: PreHookContext, name: str, result: Any
    ) -> Any:
        new_result = result
        if name in self._syft_post_hooks__:
            for hook in self._syft_post_hooks__[name]:
                new_result = hook(context, name, new_result)

        if name not in self._syft_dont_wrap_attrs():
            if HOOK_ALWAYS in self._syft_post_hooks__:
                for hook in self._syft_post_hooks__[HOOK_ALWAYS]:
                    new_result = hook(context, name, new_result)
        return new_result

    def _syft_output_action_object(
        self,
        result: Any,
    ) -> Any:
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
            context_self = self.syft_action_data  # type: ignore

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
                if context.action is not None:
                    result.syft_history_hash = context.action.syft_history_hash

            def __wrapper__bool__() -> bool:
                return result

            return __wrapper__bool__

        if self.syft_is_property(context_self, name):
            if show_print:
                print("Property detected: ", name)
            if self.syft_is_property(context_self, name):
                context = PreHookContext(obj=self, op_name=name)
                context, _, _ = self._syft_run_pre_hooks__(context, name, (), {})
                # no input needs to propagate
                result = self._syft_run_post_hooks__(
                    context, name, self.syft_get_property(context_self, name)
                )
                if name not in self._syft_dont_wrap_attrs():
                    result = self._syft_output_action_object(result)
                    if context.action is not None:
                        result.syft_history_hash = context.action.syft_history_hash
                return result

        # check for other types that aren't methods, functions etc
        def fake_func(*args: Any, **kwargs: Any) -> Any:
            return ActionDataEmpty(syft_internal_type=self.syft_internal_type)

        if (
            isinstance(self.syft_action_data, ActionDataEmpty)
            and name not in action_data_empty_must_run
        ):
            original_func = fake_func
        else:
            original_func = getattr(self.syft_action_data, name)

        if show_print:
            debug_original_func(name, original_func)
        if inspect.ismethod(original_func) or inspect.ismethoddescriptor(original_func):
            if show_print:
                print("Running method: ", name)

            def wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
                context = PreHookContext(obj=self, op_name=name)
                context, pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    context, name, args, kwargs
                )

                if not has_action_data_empty(args=args, kwargs=kwargs):
                    original_args, original_kwargs = debox_args_and_kwargs(
                        pre_hook_args, pre_hook_kwargs
                    )

                    result = original_func(*original_args, **original_kwargs)
                else:
                    result = fake_func(*args, **kwargs)

                post_result = self._syft_run_post_hooks__(context, name, result)
                if name not in self._syft_dont_wrap_attrs():
                    post_result = self._syft_output_action_object(
                        post_result,
                    )
                    if context.action is not None:
                        post_result.syft_history_hash = context.action.syft_history_hash
                    post_result.syft_node_uid = context.node_uid
                    if context.result_id is not None:
                        post_result.id = context.result_id
                return post_result

            wrapper = types.MethodType(wrapper, type(self))
        else:
            if show_print:
                print("Running non-method: ", name)

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                context = PreHookContext(obj=self, op_name=name)
                context, pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                    context, name, args, kwargs
                )

                if not has_action_data_empty(args=args, kwargs=kwargs):
                    original_args, original_kwargs = debox_args_and_kwargs(
                        pre_hook_args, pre_hook_kwargs
                    )

                    result = original_func(*original_args, **original_kwargs)
                else:
                    result = fake_func(*args, **kwargs)

                post_result = self._syft_run_post_hooks__(context, name, result)
                if name not in self._syft_dont_wrap_attrs():
                    post_result = self._syft_output_action_object(post_result)
                    if context.action is not None:
                        post_result.syft_history_hash = context.action.syft_history_hash
                    post_result.syft_node_uid = context.node_uid
                    if context.result_id is not None:
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
        from ...client.api import APIRegistry
        from ...client.api import SyftAPICall

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
        remote_self: Optional[Union[UID, LineageID]] = None,
        args: Optional[List[Union[UID, ActionObjectPointer, LineageID]]] = None,
        kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer, LineageID]]] = None,
    ) -> Action:
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        arg_ids = [
            LineageID(uid) if isinstance(uid, (UID, LineageID)) else uid.syft_lineage_id
            for uid in args
        ]
        kwarg_ids = {
            k: LineageID(uid)
            if isinstance(uid, (LineageID, UID))
            else uid.syft_lineage_id
            for k, uid in kwargs.items()
        }
        action = Action(
            path=path,
            op=op,
            remote_self=LineageID(remote_self),
            args=arg_ids,
            kwargs=kwarg_ids,
        )
        return action

    def syft_make_method_action(
        self,
        op: str,
        args: Optional[List[Union[UID, ActionObjectPointer]]] = None,
        kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]] = None,
    ) -> Action:
        path = self.syft_get_path()
        return self.syft_make_action(
            path=path, op=op, remote_self=self.syft_lineage_id, args=args, kwargs=kwargs
        )

    def syft_get_path(self) -> str:
        if isinstance(self, AnyActionObject) and self.syft_internal_type:
            return f"{self.syft_internal_type.__name__}"
        return f"{type(self).__name__}"

    def syft_remote_method(
        self,
        op: str,
    ) -> Callable:
        def wrapper(
            *args: Optional[List[Union[UID, ActionObjectPointer]]],
            **kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]],
        ) -> Action:
            return self.syft_make_method_action(op=op, args=args, kwargs=kwargs)

        return wrapper

    def keys(self) -> KeysView[str]:
        return self.syft_action_data.keys()  # type: ignore

    ###### __DUNDER_MIFFLIN__

    # if we do not implement these boiler plate __method__'s then special infix
    # operations like x + y won't trigger __getattribute__
    # unless there is a super special reason we should write no code in these functions

    def __repr__(self) -> str:
        return self.__repr__()

    def __str__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.__len__()

    def __getitem__(self, key: Any) -> Any:
        return self._syft_output_action_object(self.__getitem__(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        return self.__setitem__(key, value)

    def __contains__(self, key: Any) -> bool:
        return self.__contains__(key)

    def __bool__(self) -> bool:
        return self.__bool__()

    def __add__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__add__(other))

    def __sub__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__sub__(other))

    def __mul__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__mul__(other))

    def __matmul__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__matmul__(other))

    def __eq__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__eq__(other))

    def __lt__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__lt__(other))

    def __gt__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__gt__(other))

    def __le__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__le__(other))

    def __ge__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__ge__(other))

    def __delattr__(self, key: Any) -> None:
        self.__delattr__(key)

    def __delitem__(self, key: Any) -> None:
        self.__delitem__(key)

    def __invert__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__invert__(other))

    def __divmod__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__divmod__(other))

    def __floordiv__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__floordiv__(other))

    def __mod__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__mod__(other))

    # r ops
    # we want the underlying implementation so we should just call into __getattribute__
    def __radd__(self, other: Any) -> Any:
        return self.__radd__(other)

    def __rsub__(self, other: Any) -> Any:
        return self.__rsub__(other)

    def __rmatmul__(self, other: Any) -> Any:
        return self.__rmatmul__(other)


@serializable()
class AnyActionObject(ActionObject):
    __canonical_name__ = "AnyActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[Type[Any]] = Any  # type: ignore
    syft_passthrough_attrs: List[str] = []
    syft_dont_wrap_attrs: List[str] = []

    def __float__(self) -> float:
        return float(self.syft_action_data)


action_types[Any] = AnyActionObject
