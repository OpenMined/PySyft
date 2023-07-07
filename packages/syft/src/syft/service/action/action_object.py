# future
from __future__ import annotations

# stdlib
from enum import Enum
import inspect
import traceback
import types
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# third party
import pydantic
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ...client.api import SyftAPI
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...store.linked_obj import LinkedObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject
from ...types.syft_object import SyftObject
from ...types.uid import LineageID
from ...types.uid import UID
from ...util.logger import debug
from ..response import SyftException
from .action_data_empty import ActionDataEmpty
from .action_permissions import ActionPermission
from .action_types import action_type_for_object
from .action_types import action_type_for_type
from .action_types import action_types

NoneType = type(None)


@serializable()
class TwinMode(Enum):
    NONE = 0
    PRIVATE = 1
    MOCK = 2


@serializable()
class ActionType(Enum):
    GETATTRIBUTE = 1
    METHOD = 2
    SETATTRIBUTE = 4
    FUNCTION = 8
    CREATEOBJECT = 16


@serializable()
class Action(SyftObject):
    """Serializable Action object.

    Parameters:
        path: str
            The path of the Type of the remote object.
        op: str
            The method to be executed from the remote object.
        remote_self: Optional[LineageID]
            The extended UID of the SyftObject
        args: List[LineageID]
            `op` args
        kwargs: Dict[str, LineageID]
            `op` kwargs
        result_id: Optional[LineageID]
            Extended UID of the resulted SyftObject
    """

    __canonical_name__ = "Action"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__: List[str] = []

    path: str
    op: str
    remote_self: Optional[LineageID]
    args: List[LineageID]
    kwargs: Dict[str, LineageID]
    result_id: Optional[LineageID]
    action_type: Optional[ActionType]
    create_object: Optional[SyftObject] = None

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        """Generate or reuse an UID"""
        return v if isinstance(v, UID) else UID()

    @pydantic.validator("result_id", pre=True, always=True)
    def make_result_id(cls, v: Optional[Union[UID, LineageID]]) -> UID:
        """Generate or reuse a LineageID"""
        return v if isinstance(v, LineageID) else LineageID(v)

    @property
    def full_path(self) -> str:
        """Action path and operation"""
        return f"{self.path}.{self.op}"

    @property
    def syft_history_hash(self) -> int:
        """Create a unique hash for the operations applied on the object."""
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

    def __repr__(self):
        def repr_uid(_id):
            return f"{str(_id)[:3]}..{str(_id)[-1]}"

        arg_repr = ", ".join([repr_uid(x) for x in self.args])
        kwargs_repr = ", ".join(
            [f"{key}={repr_uid(value)}" for key, value in self.kwargs.items()]
        )
        _coll_repr_ = (
            f"[{repr_uid(self.remote_self)}]" if self.remote_self is not None else ""
        )
        return (
            f"ActionObject {self.path}{_coll_repr_}.{self.op}({arg_repr},{kwargs_repr})"
        )


class ActionObjectPointer:
    pass


# Hooks
HOOK_ALWAYS = "ALWAYS"
HOOK_ON_POINTERS = "ON_POINTERS"

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
    "get",  # syft
    "delete_data",  # syft
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
    "__str__",
]


class PreHookContext(SyftBaseObject):
    """Hook context

    Parameters:
        obj: Any
            The ActionObject to use for the action
        op_name: str
            The method name to use for the action
        node_uid: Optional[UID]
            Optional Syft node UID
        result_id: Optional[Union[UID, LineageID]]
            Optional result Syft UID
        action: Optional[Action]
            The action generated by the current hook
    """

    obj: Any
    op_name: str
    node_uid: Optional[UID]
    result_id: Optional[Union[UID, LineageID]]
    result_twin_type: Optional[TwinMode]
    action: Optional[Action]
    action_type: Optional[ActionType]


def make_action_side_effect(
    context: PreHookContext, *args: Any, **kwargs: Any
) -> Result[Ok[Tuple[PreHookContext, Tuple[Any, ...], Dict[str, Any]]], Err[str]]:
    """Create a new action from context_op_name, and add it to the PreHookContext

    Parameters:
        context: PreHookContext
            PreHookContext object
        *args:
            Operation *args
        **kwargs
            Operation *kwargs
    Returns:
        - Ok[[Tuple[PreHookContext, Tuple[Any, ...], Dict[str, Any]]] on success
        - Err[str] on failure
    """
    # relative

    try:
        action = context.obj.syft_make_action_with_self(
            op=context.op_name,
            args=args,
            kwargs=kwargs,
            action_type=context.action_type,
        )
        context.action = action
    except Exception:
        print(f"make_action_side_effect failed with {traceback.format_exc()}")
        return Err(f"make_action_side_effect failed with {traceback.format_exc()}")
    return Ok((context, args, kwargs))


class TraceResult:
    result = []
    _client = None

    @classmethod
    def reset(cls):
        cls.result = []
        cls._client = None


def trace_action_side_effect(
    context: PreHookContext, *args: Any, **kwargs: Any
) -> Result[Ok[Tuple[PreHookContext, Tuple[Any, ...], Dict[str, Any]]], Err[str]]:
    action = context.action
    if action is not None:
        TraceResult.result += [action]
    return Ok((context, args, kwargs))


def convert_to_pointers(
    api: SyftAPI,
    node_uid: Optional[UID] = None,
    args: Optional[List] = None,
    kwargs: Optional[Dict] = None,
) -> Tuple[List, Dict]:
    arg_list = []
    kwarg_dict = {}
    if args is not None:
        for arg in args:
            if not isinstance(arg, ActionObject):
                arg = ActionObject.from_obj(arg)
                arg.syft_node_uid = node_uid
                arg = api.services.action.set(arg)
                # arg = action_obj.send(
                #     client
                # )  # make sure this doesn't break things later on in send_method_action
            arg_list.append(arg)

    if kwargs is not None:
        for k, arg in kwargs.items():
            if not isinstance(arg, ActionObject):
                arg = ActionObject.from_obj(arg)
                arg.syft_node_uid = node_uid
                arg = api.services.action.set(arg)
                # arg = action_obj.send(client)

            kwarg_dict[k] = arg

    return arg_list, kwarg_dict


def send_action_side_effect(
    context: PreHookContext, *args: Any, **kwargs: Any
) -> Result[Ok[Tuple[PreHookContext, Tuple[Any, ...], Dict[str, Any]]], Err[str]]:
    """Create a new action from the context.op_name, and execute it on the remote node."""
    try:
        if context.action is None:
            result = make_action_side_effect(context, *args, **kwargs)
            if result.is_err():
                raise RuntimeError(result.err())

            context, _, _ = result.ok()

        action_result = context.obj.syft_execute_action(context.action, sync=True)

        if not isinstance(action_result, ActionObject):
            raise RuntimeError(f"Got back unexpected response : {action_result}")
        else:
            context.node_uid = action_result.syft_node_uid
            context.result_id = action_result.id
            context.result_twin_type = action_result.syft_twin_type
    except Exception as e:
        return Err(
            f"send_action_side_effect failed with {e}\n {traceback.format_exc()}"
        )
    return Ok((context, args, kwargs))


def propagate_node_uid(
    context: PreHookContext, op: str, result: Any
) -> Result[Ok[Any], Err[str]]:
    """Patch the result to include the syft_node_uid

    Parameters:
        context: PreHookContext
            PreHookContext object
        op: str
            Which operation was executed
        result: Any
            The result to patch
    Returns:
        - Ok[[result] on success
        - Err[str] on failure
    """
    if context.op_name in dont_make_side_effects or not hasattr(
        context.obj, "syft_node_uid"
    ):
        return Ok(result)

    try:
        syft_node_uid = getattr(context.obj, "syft_node_uid", None)
        if syft_node_uid is None:
            raise RuntimeError("Can't proagate node_uid because parent doesnt have one")

        if op not in context.obj._syft_dont_wrap_attrs():
            if hasattr(result, "syft_node_uid"):
                result.syft_node_uid = syft_node_uid
        else:
            raise RuntimeError("dont propogate node_uid because output isnt wrapped")
    except Exception:
        return Err(f"propagate_node_uid failed with {traceback.format_exc()}")

    return Ok(result)


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


BASE_PASSTHROUGH_ATTRS = [
    "is_mock",
    "is_real",
    "is_twin",
    "is_pointer",
    "request",
    "__repr__",
    "_repr_markdown_",
    "syft_twin_type",
    "_repr_debug_",
    "as_empty",
    "get",
]


class ActionObject(SyftObject):
    """Action object for remote execution."""

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
    _syft_pre_hooks__: Dict[str, List] = {}
    _syft_post_hooks__: Dict[str, List] = {}
    syft_twin_type: TwinMode = TwinMode.NONE
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS
    # syft_dont_wrap_attrs = ["shape"]

    @property
    def is_pointer(self) -> bool:
        return self.syft_node_uid is not None

    @property
    def syft_lineage_id(self) -> LineageID:
        """Compute the LineageID of the ActionObject, using the `id` and the `syft_history_hash` memebers"""
        return LineageID(self.id, self.syft_history_hash)

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        """Generate or reuse an UID"""
        return Action.make_id(v)

    @property
    def is_mock(self):
        return self.syft_twin_type == TwinMode.MOCK

    @property
    def is_real(self):
        return self.syft_twin_type == TwinMode.PRIVATE

    @property
    def is_twin(self):
        return self.syft_twin_type != TwinMode.NONE

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

    def syft_point_to(self, node_uid: UID) -> "ActionObject":
        """Set the syft_node_uid, used in the post hooks"""
        self.syft_node_uid = node_uid
        return self

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

    def syft_execute_action(
        self, action: Action, sync: bool = True
    ) -> ActionObjectPointer:
        """Execute a remote action

        Parameters:
            action: Action
                Which action to execute
            sync: bool
                Run sync/async

        Returns:
            ActionObjectPointer
        """
        if self.syft_node_uid is None:
            raise SyftException("Pointers can't execute without a node_uid.")

        # relative
        from ...client.api import APIRegistry
        from ...client.api import SyftAPICall

        api = APIRegistry.api_for(
            node_uid=self.syft_node_uid,
            user_verify_key=self.syft_client_verify_key,
        )

        kwargs = {"action": action}
        api_call = SyftAPICall(
            node_uid=self.syft_node_uid, path="action.execute", args=[], kwargs=kwargs
        )
        return api.make_call(api_call)

    def request(self, client):
        # relative
        from ..request.request import ActionStoreChange
        from ..request.request import SubmitRequest

        action_object_link = LinkedObject.from_obj(self, node_uid=self.syft_node_uid)
        permission_change = ActionStoreChange(
            linked_obj=action_object_link, apply_permission_type=ActionPermission.READ
        )

        submit_request = SubmitRequest(
            changes=[permission_change],
            requesting_user_verify_key=client.credentials.verify_key,
        )
        return client.api.services.request.submit(submit_request)

    def _syft_try_to_save_to_store(self, obj) -> None:
        if self.syft_node_uid is None or self.syft_client_verify_key is None:
            return
        elif obj.syft_node_uid is not None:
            return
        # TODO fix: the APIRegistry often gets the wrong client
        # if you have 2 clients in memory
        # therefore the following happens if you call a method
        # with a pointer to a twin (mock version)
        # 1) it gets the wrong credentials
        # 2) it uses the mock version to overwrite the real version
        # 3) it shouldnt send in the first place as it already exists

        # relative
        from ...client.api import APIRegistry

        action = Action(
            path="",
            op="",
            remote_self=None,
            result_id=obj.id,
            args=[],
            kwargs=dict(),
            action_type=ActionType.CREATEOBJECT,
            create_object=obj,
        )

        if TraceResult._client is not None:
            api = TraceResult._client.api
            TraceResult.result += [action]
        else:
            api = APIRegistry.api_for(
                node_uid=self.syft_node_uid,
                user_verify_key=self.syft_client_verify_key,
            )
        api.services.action.execute(action)

    def _syft_prepare_obj_uid(self, obj) -> LineageID:
        # We got the UID
        if isinstance(obj, (UID, LineageID)):
            return LineageID(obj.id)

        # We got the ActionObjectPointer
        if isinstance(obj, ActionObjectPointer):
            return obj.syft_lineage_id

        # We got the ActionObject. We need to save it in the store.
        if isinstance(obj, ActionObject):
            self._syft_try_to_save_to_store(obj)
            return obj.syft_lineage_id

        # We got a raw object. We need to create the ActionObject from scratch and save it in the store.
        obj_id = Action.make_id(None)
        lin_obj_id = Action.make_result_id(obj_id)
        act_obj = ActionObject.from_obj(obj, id=obj_id, syft_lineage_id=lin_obj_id)

        self._syft_try_to_save_to_store(act_obj)

        return act_obj.syft_lineage_id

    def syft_make_action(
        self,
        path: str,
        op: str,
        remote_self: Optional[Union[UID, LineageID]] = None,
        args: Optional[
            List[Union[UID, LineageID, ActionObjectPointer, ActionObject, Any]]
        ] = None,
        kwargs: Optional[
            Dict[str, Union[UID, LineageID, ActionObjectPointer, ActionObject, Any]]
        ] = None,
        action_type: Optional[ActionType] = None,
    ) -> Action:
        """Generate new action from the information

        Parameters:
            path: str
                The path of the Type of the remote object.
            op: str
                The method to be executed from the remote object.
            remote_self: Optional[Union[UID, LineageID]]
                The extended UID of the SyftObject
            args: Optional[List[Union[UID, LineageID, ActionObjectPointer, ActionObject]]]
                `op` args
            kwargs: Optional[Dict[str, Union[UID, LineageID, ActionObjectPointer, ActionObject]]]
                `op` kwargs
        Returns:
            Action object

        Raises:
            ValueError: For invalid args or kwargs
            PydanticValidationError: For args and kwargs
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        arg_ids = []
        kwarg_ids = {}

        for obj in args:
            arg_ids.append(self._syft_prepare_obj_uid(obj))

        for k, obj in kwargs.items():
            kwarg_ids[k] = self._syft_prepare_obj_uid(obj)

        action = Action(
            path=path,
            op=op,
            remote_self=LineageID(remote_self),
            args=arg_ids,
            kwargs=kwarg_ids,
            action_type=action_type,
        )
        return action

    def syft_make_action_with_self(
        self,
        op: str,
        args: Optional[List[Union[UID, ActionObjectPointer]]] = None,
        kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]] = None,
        action_type: Optional[ActionType] = None,
    ) -> Action:
        """Generate new method action from the current object.

        Parameters:
            op: str
                The method to be executed from the remote object.
            args: List[LineageID]
                `op` args
            kwargs: Dict[str, LineageID]
                `op` kwargs
        Returns:
            Action object

        Raises:
            ValueError: For invalid args or kwargs
            PydanticValidationError: For args and kwargs
        """
        path = self.syft_get_path()
        return self.syft_make_action(
            path=path,
            op=op,
            remote_self=self.syft_lineage_id,
            args=args,
            kwargs=kwargs,
            action_type=action_type,
        )

    def syft_get_path(self) -> str:
        """Get the type path of the underlying object"""
        if isinstance(self, AnyActionObject) and self.syft_internal_type:
            return f"{type(self.syft_action_data).__name__}"  # avoids AnyActionObject errors
        return f"{type(self).__name__}"

    def syft_remote_method(
        self,
        op: str,
    ) -> Callable:
        """Generate a Callable object for remote calls.

        Parameters:
            op: str
                he method to be executed from the remote object.

        Returns:
            A function
        """

        def wrapper(
            *args: Optional[List[Union[UID, ActionObjectPointer]]],
            **kwargs: Optional[Dict[str, Union[UID, ActionObjectPointer]]],
        ) -> Action:
            return self.syft_make_action_with_self(op=op, args=args, kwargs=kwargs)

        return wrapper

    def send(self, client: SyftClient) -> Self:
        """Send the object to a Syft Client"""
        res = client.api.services.action.set(self)
        res.syft_node_location = client.id
        res.syft_client_verify_key = client.verify_key
        return res

    def get_from(self, client: SyftClient) -> Any:
        """Get the object from a Syft Client"""
        res = client.api.services.action.get(self.id)
        if not isinstance(res, ActionObject):
            return SyftError(message=f"{res}")
        else:
            return res.syft_action_data

    def get(self) -> Any:
        """Get the object from a Syft Client"""
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        res = api.services.action.get(self.id)

        if not isinstance(res, ActionObject):
            return SyftError(message=f"{res}")
        else:
            return res.syft_action_data

    def as_empty(self):
        id = self.id
        # TODO: fix
        if isinstance(id, LineageID):
            id = id.id
        return ActionObject.empty(self.syft_internal_type, id, self.syft_lineage_id)

    @staticmethod
    def from_obj(
        syft_action_data: Any,
        id: Optional[UID] = None,
        syft_lineage_id: Optional[LineageID] = None,
    ) -> ActionObject:
        """Create an ActionObject from an existing object.

        Parameters:
            syft_action_data: Any
                The object to be converted to a Syft ActionObject
            id: Optional[UID]
                Which ID to use for the ActionObject. Optional
            syft_lineage_id: Optional[LineageID]
                Which LineageID to use for the ActionObject. Optional
        """
        if id and syft_lineage_id and id != syft_lineage_id.id:
            raise ValueError("UID and LineageID should match")

        action_type = action_type_for_object(syft_action_data)
        action_object = action_type(syft_action_data=syft_action_data)

        if id:
            action_object.id = id

        if syft_lineage_id:
            action_object.id = syft_lineage_id.id
            action_object.syft_history_hash = syft_lineage_id.syft_history_hash
        elif id:
            action_object.syft_history_hash = hash(id)

        return action_object

    @classmethod
    def add_trace_hook(cls):
        return True
        # if trace_action_side_effect not in self._syft_pre_hooks__[HOOK_ALWAYS]:
        #     self._syft_pre_hooks__[HOOK_ALWAYS].append(trace_action_side_effect)

    @classmethod
    def remove_trace_hook(cls):
        return True
        # self._syft_pre_hooks__[HOOK_ALWAYS].pop(trace_action_side_effct, None)

    @staticmethod
    def empty(
        syft_internal_type: Type[Any] = NoneType,
        id: Optional[UID] = None,
        syft_lineage_id: Optional[LineageID] = None,
    ) -> ActionObject:
        """Create an ActionObject from a type, using a ActionDataEmpty object

        Parameters:
            syft_internal_type: Type
                The Type for which to create a ActionDataEmpty object
            id: Optional[UID]
                Which ID to use for the ActionObject. Optional
            syft_lineage_id: Optional[LineageID]
                Which LineageID to use for the ActionObject. Optional
        """

        empty = ActionDataEmpty(syft_internal_type=syft_internal_type)
        res = ActionObject.from_obj(
            syft_action_data=empty, id=id, syft_lineage_id=syft_lineage_id
        )
        res.__dict__["syft_internal_type"] = syft_internal_type
        return res

    def delete_data(self):
        empty = ActionDataEmpty(syft_internal_type=self.syft_internal_type)
        self.syft_action_data = empty

    def __post_init__(self) -> None:
        """Add pre/post hooks."""
        if HOOK_ALWAYS not in self._syft_pre_hooks__:
            self._syft_pre_hooks__[HOOK_ALWAYS] = []

        if HOOK_ON_POINTERS not in self._syft_post_hooks__:
            self._syft_pre_hooks__[HOOK_ON_POINTERS] = []

        # this should be a list as orders matters
        for side_effect in [make_action_side_effect]:
            if side_effect not in self._syft_pre_hooks__[HOOK_ALWAYS]:
                self._syft_pre_hooks__[HOOK_ALWAYS].append(side_effect)

        for side_effect in [send_action_side_effect]:
            if side_effect not in self._syft_pre_hooks__[HOOK_ON_POINTERS]:
                self._syft_pre_hooks__[HOOK_ON_POINTERS].append(side_effect)

        if trace_action_side_effect not in self._syft_pre_hooks__[HOOK_ALWAYS]:
            self._syft_pre_hooks__[HOOK_ALWAYS].append(trace_action_side_effect)

        if HOOK_ALWAYS not in self._syft_post_hooks__:
            self._syft_post_hooks__[HOOK_ALWAYS] = []

        if HOOK_ON_POINTERS not in self._syft_post_hooks__:
            self._syft_post_hooks__[HOOK_ON_POINTERS] = []

        for side_effect in [propagate_node_uid]:
            if side_effect not in self._syft_post_hooks__[HOOK_ALWAYS]:
                self._syft_post_hooks__[HOOK_ALWAYS].append(side_effect)

        if isinstance(self.syft_action_data, ActionObject):
            raise Exception("Nested ActionObjects", self.syft_action_data)

        self.syft_history_hash = hash(self.id)

    def _syft_run_pre_hooks__(
        self, context: PreHookContext, name: str, args: Any, kwargs: Any
    ) -> Tuple[PreHookContext, Tuple[Any, ...], Dict[str, Any]]:
        """Hooks executed before the actual call"""
        result_args, result_kwargs = args, kwargs
        if name in self._syft_pre_hooks__:
            for hook in self._syft_pre_hooks__[name]:
                result = hook(context, *result_args, **result_kwargs)
                if result.is_ok():
                    context, result_args, result_kwargs = result.ok()
                else:
                    debug(f"Pre-hook failed with {result.err()}")
        if name not in self._syft_dont_wrap_attrs():
            if HOOK_ALWAYS in self._syft_pre_hooks__:
                for hook in self._syft_pre_hooks__[HOOK_ALWAYS]:
                    result = hook(context, *result_args, **result_kwargs)
                    if result.is_ok():
                        context, result_args, result_kwargs = result.ok()
                    else:
                        msg = result.err().replace("\\n", "\n")
                        debug(f"Pre-hook failed with {msg}")

        if self.is_pointer:
            if name not in self._syft_dont_wrap_attrs():
                if HOOK_ALWAYS in self._syft_pre_hooks__:
                    for hook in self._syft_pre_hooks__[HOOK_ON_POINTERS]:
                        result = hook(context, *result_args, **result_kwargs)
                        if result.is_ok():
                            context, result_args, result_kwargs = result.ok()
                        else:
                            msg = result.err().replace("\\n", "\n")
                            debug(f"Pre-hook failed with {msg}")

        return context, result_args, result_kwargs

    def _syft_run_post_hooks__(
        self, context: PreHookContext, name: str, result: Any
    ) -> Any:
        """Hooks executed after the actual call"""
        new_result = result
        if name in self._syft_post_hooks__:
            for hook in self._syft_post_hooks__[name]:
                result = hook(context, name, new_result)
                if result.is_ok():
                    new_result = result.ok()
                else:
                    debug(f"Post hook failed with {result.err()}")

        if name not in self._syft_dont_wrap_attrs():
            if HOOK_ALWAYS in self._syft_post_hooks__:
                for hook in self._syft_post_hooks__[HOOK_ALWAYS]:
                    result = hook(context, name, new_result)
                    if result.is_ok():
                        new_result = result.ok()
                    else:
                        debug(f"Post hook failed with {result.err()}")

        if self.is_pointer:
            if name not in self._syft_dont_wrap_attrs():
                if HOOK_ALWAYS in self._syft_post_hooks__:
                    for hook in self._syft_post_hooks__[HOOK_ON_POINTERS]:
                        result = hook(context, name, new_result)
                        if result.is_ok():
                            new_result = result.ok()
                        else:
                            debug(f"Post hook failed with {result.err()}")

        return new_result

    def _syft_output_action_object(
        self, result: Any, context: Optional[PreHookContext] = None
    ) -> Any:
        """Wrap the result in an ActionObject"""
        if issubclass(type(result), ActionObject):
            return result

        constructor = action_type_for_type(result)
        syft_twin_type = TwinMode.NONE
        if context.result_twin_type is not None:
            syft_twin_type = context.result_twin_type
        result = constructor(syft_action_data=result, syft_twin_type=syft_twin_type)

        return result

    def _syft_passthrough_attrs(self) -> List[str]:
        """These attributes are forwarded to the `object` base class."""
        return passthrough_attrs + getattr(self, "syft_passthrough_attrs", [])

    def _syft_dont_wrap_attrs(self) -> List[str]:
        """The results from these attributes are ignored from UID patching."""
        return dont_wrap_output_attrs + getattr(self, "syft_dont_wrap_attrs", [])

    def _syft_get_attr_context(self, name: str) -> Any:
        """Find which instance - Syft ActionObject or the original object - has the requested attribute."""
        defined_on_self = name in self.__dict__ or name in self.__private_attributes__

        debug(">> ", name, ", defined_on_self = ", defined_on_self)

        # use the custom defined version
        context_self = self
        if not defined_on_self:
            context_self = self.syft_action_data  # type: ignore

        return context_self

    def _syft_attr_propagate_ids(self, context, name: str, result: Any) -> Any:
        """Patch the results with the syft_history_hash, node_uid, and result_id."""
        if name in self._syft_dont_wrap_attrs():
            return result

        # Wrap as Syft Object
        result = self._syft_output_action_object(result, context)

        # Propagate History
        if context.action is not None:
            result.syft_history_hash = context.action.syft_history_hash

        # Propagate Syft Node UID
        result.syft_node_uid = context.node_uid

        # Propogate Syft Node Location and Client Verify Key
        result.syft_node_location = context.syft_node_location
        result.syft_client_verify_key = context.syft_client_verify_key

        # Propagate Result ID
        if context.result_id is not None:
            result.id = context.result_id

        return result

    def _syft_wrap_attribute_for_bool_on_nonbools(self, name: str) -> Any:
        """Handle `__getattribute__` for bool casting."""
        if name != "__bool__":
            raise RuntimeError(
                "[_wrap_attribute_for_bool_on_nonbools] Use this only for the __bool__ operator"
            )

        if hasattr(self.syft_action_data, "__bool__"):
            raise RuntimeError(
                "[_wrap_attribute_for_bool_on_nonbools] self.syft_action_data already implements the bool operator"
            )

        debug("[__getattribute__] Handling bool on nonbools")
        context = PreHookContext(
            obj=self,
            op_name=name,
            syft_node_location=self.syft_node_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        context, _, _ = self._syft_run_pre_hooks__(context, name, (), {})

        # no input needs to propagate
        result = self._syft_run_post_hooks__(context, name, bool(self.syft_action_data))
        result = self._syft_attr_propagate_ids(context, name, result)

        def __wrapper__bool__() -> bool:
            return result

        return __wrapper__bool__

    def _syft_wrap_attribute_for_properties(self, name: str) -> Any:
        """Handle `__getattribute__` for properties."""
        context_self = self._syft_get_attr_context(name)

        if not self.syft_is_property(context_self, name):
            raise RuntimeError(
                "[_wrap_attribute_for_properties] Use this only on properties"
            )
        debug(f"[__getattribute__] Handling property {name} ")

        context = PreHookContext(
            obj=self,
            op_name=name,
            action_type=ActionType.GETATTRIBUTE,
            syft_node_location=self.syft_node_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        context, _, _ = self._syft_run_pre_hooks__(context, name, (), {})
        # no input needs to propagate
        result = self._syft_run_post_hooks__(
            context, name, self.syft_get_property(context_self, name)
        )

        return self._syft_attr_propagate_ids(context, name, result)

    def _syft_wrap_attribute_for_methods(self, name: str) -> Any:
        """Handle `__getattribute__` for methods."""

        # check for other types that aren't methods, functions etc
        def fake_func(*args: Any, **kwargs: Any) -> Any:
            return ActionDataEmpty(syft_internal_type=self.syft_internal_type)

        debug(f"[__getattribute__] Handling method {name} ")
        if (
            isinstance(self.syft_action_data, ActionDataEmpty)
            and name not in action_data_empty_must_run
        ):
            original_func = fake_func
        else:
            original_func = getattr(self.syft_action_data, name)

        debug_original_func(name, original_func)

        def _base_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = PreHookContext(
                obj=self,
                op_name=name,
                action_type=ActionType.METHOD,
                syft_node_location=self.syft_node_location,
                syft_client_verify_key=self.syft_client_verify_key,
            )
            context, pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
                context, name, args, kwargs
            )

            if has_action_data_empty(args=args, kwargs=kwargs):
                result = fake_func(*args, **kwargs)
            else:
                original_args, original_kwargs = debox_args_and_kwargs(
                    pre_hook_args, pre_hook_kwargs
                )
                result = original_func(*original_args, **original_kwargs)

            post_result = self._syft_run_post_hooks__(context, name, result)
            post_result = self._syft_attr_propagate_ids(context, name, post_result)

            return post_result

        if inspect.ismethod(original_func) or inspect.ismethoddescriptor(original_func):
            debug("Running method: ", name)

            def wrapper(_self: Any, *args: Any, **kwargs: Any):
                return _base_wrapper(*args, **kwargs)

            wrapper = types.MethodType(wrapper, type(self))
        else:
            debug("Running non-method: ", name)

            wrapper = _base_wrapper

        try:
            wrapper.__doc__ = original_func.__doc__
            debug(
                "Found original signature for ",
                name,
                inspect.signature(original_func),
            )
            wrapper.__ipython_inspector_signature_override__ = inspect.signature(
                original_func
            )
        except Exception:
            debug("name", name, "has no signature")

        return wrapper

    def _syft_setattr(self, name, value):
        args = (name, value)
        kwargs = dict()
        op_name = "__setattr__"

        def fake_func(*args: Any, **kwargs: Any) -> Any:
            return ActionDataEmpty(syft_internal_type=self.syft_internal_type)

        if isinstance(self.syft_action_data, ActionDataEmpty) or has_action_data_empty(
            args=args, kwargs=kwargs
        ):
            local_func = fake_func
        else:
            local_func = getattr(self.syft_action_data, op_name)

        context = PreHookContext(
            obj=self,
            op_name=op_name,
            action_type=ActionType.SETATTRIBUTE,
            syft_node_location=self.syft_node_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        context, pre_hook_args, pre_hook_kwargs = self._syft_run_pre_hooks__(
            context, "__setattr__", args, kwargs
        )

        original_args, _ = debox_args_and_kwargs(pre_hook_args, pre_hook_kwargs)
        val = original_args[1]
        local_func(name, val)
        local_result = self

        post_result = self._syft_run_post_hooks__(context, op_name, local_result)
        post_result = self._syft_attr_propagate_ids(context, op_name, post_result)
        return post_result

    def __getattribute__(self, name: str) -> Any:
        """Called unconditionally to implement attribute accesses for instances of the class.
        If the class also defines __getattr__(), the latter will not be called unless __getattribute__()
        either calls it explicitly or raises an AttributeError.
        This method should return the (computed) attribute value or raise an AttributeError exception.
        In order to avoid infinite recursion in this method, its implementation should always:
         * call the base class method with the same name to access any attributes it needs
            for example : object.__getattribute__(self, name).
         * use the syft/_syft prefix for internal methods.
         * add the method name to the passthrough_attrs.

        Parameters:
            name: str
                The name of the attribute to access.
        """
        # bypass certain attrs to prevent recursion issues
        if name.startswith("_syft") or name.startswith("syft"):
            return object.__getattribute__(self, name)

        # third party
        if name in self._syft_passthrough_attrs():
            return object.__getattribute__(self, name)
        context_self = self._syft_get_attr_context(name)

        # Handle bool operator on nonbools
        if name == "__bool__" and not hasattr(self.syft_action_data, "__bool__"):
            return self._syft_wrap_attribute_for_bool_on_nonbools(name)

        # Handle Properties
        if self.syft_is_property(context_self, name):
            return self._syft_wrap_attribute_for_properties(name)

        # Handle anything else
        return self._syft_wrap_attribute_for_methods(name)

    def __setattr__(self, name: str, value: Any) -> Any:
        defined_on_self = name in self.__dict__ or name in self.__private_attributes__

        debug(">> ", name, ", defined_on_self = ", defined_on_self)

        # use the custom defined version
        if defined_on_self:
            self.__dict__[name] = value
            return value
        else:
            self._syft_setattr(name, value)
            context_self = self.syft_action_data  # type: ignore
            return context_self.__setattr__(name, value)

    def keys(self) -> KeysView[str]:
        return self.syft_action_data.keys()  # type: ignore

    ###### __DUNDER_MIFFLIN__

    # if we do not implement these boiler plate __method__'s then special infix
    # operations like x + y won't trigger __getattribute__
    # unless there is a super special reason we should write no code in these functions
    def _repr_markdown_(self) -> str:
        if self.is_mock:
            res = "TwinPointer(Mock)"
        elif self.is_real:
            res = "TwinPointer(Real)"
        elif not self.is_twin:
            res = "Pointer"
        child_repr = (
            self.syft_action_data._repr_markdown_()
            if hasattr(self.syft_action_data, "_repr_markdown_")
            else self.syft_action_data.__repr__()
        )

        return f"```python\n{res}\n```\n{child_repr}"

    def __repr__(self) -> str:
        if self.is_mock:
            res = "TwinPointer(Mock)"
        elif self.is_real:
            res = "TwinPointer(Real)"
        if not self.is_twin:
            res = "Pointer"
        return f"{res}:\n{str(self.syft_action_data)}"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__call__(*args, **kwds)

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

    def __invert__(self) -> Any:
        return self._syft_output_action_object(self.__invert__())

    def __round__(self) -> Any:
        return self._syft_output_action_object(self.__round__())

    def __pos__(self) -> Any:
        return self._syft_output_action_object(self.__pos__())

    def __trunc__(self) -> Any:
        return self._syft_output_action_object(self.__trunc__())

    def __divmod__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__divmod__(other))

    def __floordiv__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__floordiv__(other))

    def __mod__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__mod__(other))

    def __abs__(self) -> Any:
        return self._syft_output_action_object(self.__abs__())

    def __neg__(self) -> Any:
        return self._syft_output_action_object(self.__neg__())

    def __or__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__or__(other))

    def __and__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__and__(other))

    def __xor__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__xor__(other))

    def __pow__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__pow__(other))

    def __truediv__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__truediv__(other))

    def __lshift__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__lshift__(other))

    def __rshift__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rshift__(other))

    def __iter__(self):
        return self._syft_output_action_object(self.__iter__())

    def __next__(self):
        return self._syft_output_action_object(self.__next__())

    # r ops
    # we want the underlying implementation so we should just call into __getattribute__
    def __radd__(self, other: Any) -> Any:
        return self.__radd__(other)

    def __rsub__(self, other: Any) -> Any:
        return self.__rsub__(other)

    def __rmul__(self, other: Any) -> Any:
        return self.__rmul__(other)

    def __rmatmul__(self, other: Any) -> Any:
        return self.__rmatmul__(other)

    def __rmod__(self, other: Any) -> Any:
        return self.__rmod__(other)

    def __ror__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__ror__(other))

    def __rand__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rand__(other))

    def __rxor__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rxor__(other))

    def __rpow__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rpow__(other))

    def __rtruediv__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rtruediv__(other))

    def __rfloordiv__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rfloordiv__(other))

    def __rlshift__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rlshift__(other))

    def __rrshift__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__rrshift__(other))


@serializable()
class AnyActionObject(ActionObject):
    __canonical_name__ = "AnyActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[Type[Any]] = NoneType  # type: ignore
    # syft_passthrough_attrs: List[str] = []
    syft_dont_wrap_attrs: List[str] = ["__str__", "__repr__"]

    def __float__(self) -> float:
        return float(self.syft_action_data)

    def __int__(self) -> float:
        return int(self.syft_action_data)


action_types[Any] = AnyActionObject


def debug_original_func(name: str, func: Callable) -> None:
    debug(f"{name} func is:")
    debug("inspect.isdatadescriptor", inspect.isdatadescriptor(func))
    debug("inspect.isgetsetdescriptor", inspect.isgetsetdescriptor(func))
    debug("inspect.isfunction", inspect.isfunction(func))
    debug("inspect.isbuiltin", inspect.isbuiltin(func))
    debug("inspect.ismethod", inspect.ismethod(func))
    debug("inspect.ismethoddescriptor", inspect.ismethoddescriptor(func))


def is_action_data_empty(obj: Any) -> bool:
    return isinstance(obj, AnyActionObject) and isinstance(
        obj.syft_action_data, ActionDataEmpty
    )


def has_action_data_empty(args: Any, kwargs: Any) -> bool:
    for a in args:
        if is_action_data_empty(a):
            return True

    for _, a in kwargs.items():
        if is_action_data_empty(a):
            return True
    return False
