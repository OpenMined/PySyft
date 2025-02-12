# future
from __future__ import annotations

# stdlib
from collections.abc import Callable
from collections.abc import Iterable
from enum import Enum
import inspect
from io import BytesIO
import logging
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING

# third party
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

# relative
from ...client.api import SyftAPI
from ...client.api import SyftAPICall
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...serde.serialize import _serialize as serialize
from ...server.credentials import SyftVerifyKey
from ...service.blob_storage.util import can_upload_to_blob_storage
from ...service.response import SyftSuccess
from ...service.response import SyftWarning
from ...store.linked_obj import LinkedObject
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.errors import SyftException
from ...types.result import Err
from ...types.result import as_result
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject
from ...types.syft_object import SyftObject
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import LineageID
from ...types.uid import UID
from ...util.util import prompt_warning_message
from ..context import AuthedServiceContext
from ..service import from_api_or_context
from .action_data_empty import ActionDataEmpty
from .action_data_empty import ActionDataLink
from .action_data_empty import ObjectNotReady
from .action_permissions import ActionPermission
from .action_types import action_type_for_object
from .action_types import action_type_for_type
from .action_types import action_types

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # relative
    from ..sync.diff_state import AttrDiff

NoneType = type(None)


@serializable(canonical_name="TwinMode", version=1)
class TwinMode(Enum):
    NONE = 0
    PRIVATE = 1
    MOCK = 2


@serializable(canonical_name="ActionType", version=1)
class ActionType(Enum):
    GETATTRIBUTE = 1
    METHOD = 2
    SETATTRIBUTE = 4
    FUNCTION = 8
    CREATEOBJECT = 16
    SYFTFUNCTION = 32
    TWINAPI = 64


def repr_cls(c: Any) -> str:
    return f"{c.__module__}.{c.__name__}"


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

    __attr_searchable__: ClassVar[list[str]] = []

    path: str | None = None
    op: str | None = None
    remote_self: LineageID | None = None
    args: list[LineageID]
    kwargs: dict[str, LineageID]
    result_id: LineageID = Field(default_factory=lambda: LineageID(UID()))
    action_type: ActionType | None = None
    create_object: SyftObject | None = None
    user_code_id: UID | None = None

    @field_validator("result_id", mode="before")
    @classmethod
    def make_result_id(cls, v: Any) -> LineageID:
        return v if isinstance(v, LineageID) else LineageID(v)

    @property
    def full_path(self) -> str:
        """Action path and operation"""
        return f"{self.path}.{self.op}"

    @property
    def job_display_name(self) -> str:
        if self.user_code_id is not None:
            api = self.get_api()
            user_code = api.services.code.get_by_id(self.user_code_id)
            return user_code.service_func_name
        else:
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

    @classmethod
    def syft_function_action_from_kwargs_and_id(
        cls, kwargs: dict[str, Any], user_code_id: UID
    ) -> Self:
        kwarg_ids = {}
        for k, v in kwargs.items():
            kwarg_ids[k] = LineageID(v)
        return cls(
            args=[],
            kwargs=kwarg_ids,
            result_id=LineageID(),
            action_type=ActionType.SYFTFUNCTION,
            user_code_id=user_code_id,
        )

    @classmethod
    def from_api_call(cls, api_call: SyftAPICall) -> Action:
        # relative
        from ..code.user_code_service import map_kwargs_to_id

        kwargs = api_call.kwargs
        kwargs.pop("communication_protocol", None)
        function_id = kwargs.pop("uid", None)
        kwargs = map_kwargs_to_id(kwargs)
        kwarg_ids = {}
        for k, v in kwargs.items():
            kwarg_ids[k] = LineageID(v)

        action = cls(
            args=[],
            kwargs=kwarg_ids,
            result_id=LineageID(),
            action_type=ActionType.SYFTFUNCTION,
            user_code_id=function_id,
        )
        return action

    @classmethod
    def from_api_endpoint_execution(cls: type[Self]) -> Action:
        return cls(
            args=[],
            kwargs={},
            result_id=LineageID(),
            action_type=ActionType.TWINAPI,
            user_code_id=None,
        )

    def __repr__(self) -> str:
        def repr_uid(_id: LineageID) -> str:
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


@serializable(canonical_name="ActionObjectPointer", version=1)
class ActionObjectPointer:
    pass


# Hooks
HOOK_ALWAYS = "ALWAYS"
HOOK_ON_POINTERS = "ON_POINTERS"

passthrough_attrs = [
    "id",
    "refresh_object",
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
    "__pydantic_fields__",  # pydantic
    "__post_init__",  # syft
    "_get_api",  # syft
    "__validate_private_attrs__",  # syft
    "id",  # syft
    "created_date",  # syft
    "updated_date",  # syft
    "deleted_date",  # syft
    "__attr_searchable__",  # syft
    "__attr_unique__",  # syft
    "__canonical_name__",  # syft
    "__version__",  # syft
    "__args__",  # pydantic
    "to_pointer",  # syft
    "to",  # syft
    "send",  # syft
    "_send",  # syft
    "_copy_and_set_values",  # pydantic
    "get_from",  # syft
    "get",  # syft
    "delete_data",  # syft
    "_save_to_blob_storage_",  # syft
    "syft_action_data",  # syft
    "syft_resolved",  # syft
    "syft_action_data_server_id",
    "server_uid",
    "migrate_to",  # syft
    "to_dict",  # syft
    "dict",  # syft
    "has_storage_permission",  # syft
    "_iter",  # pydantic
    "__exclude_fields__",  # pydantic
    "__include_fields__",  # pydantic
    "_calculate_keys",  # pydantic
    "_get_value",  # pydantic
    "__pydantic_validator__",  # pydantic
    "__class_vars__",  # pydantic
    "__private_attributes__",  # pydantic
    "__signature__",  # pydantic
    "__pydantic_complete__",  # pydantic
    "__pydantic_core_schema__",  # pydantic
    "__pydantic_custom_init__",  # pydantic
    "__pydantic_decorators__",  # pydantic
    "__pydantic_generic_metadata__",  # pydantic
    "__pydantic_parent_namespace__",  # pydantic
    "__pydantic_post_init__",  # pydantic
    "__pydantic_root_model__",  # pydantic
    "__pydantic_serializer__",  # pydantic
    "__pydantic_validator__",  # pydantic
    "__pydantic_extra__",  # pydantic
    "__pydantic_fields_set__",  # pydantic
    "__pydantic_private__",  # pydantic
    "model_config",  # pydantic
    "model_computed_fields",  # pydantic
    "model_extra",  # pydantic
    "model_fields",  # pydantic
    "model_fields_set",  # pydantic
    "model_construct",  # pydantic
    "model_copy",  # pydantic
    "model_dump",  # pydantic
    "model_dump_json",  # pydantic
    "model_json_schema",  # pydantic
    "model_parametrized_name",  # pydantic
    "model_post_init",  # pydantic
    "model_rebuild",  # pydantic
    "model_validate",  # pydantic
    "model_validate_json",  # pydantic
    "copy",  # pydantic
    "__sha256__",  # syft
    "__hash_exclude_attrs__",  # syft
    "__private_sync_attr_mocks__",  # syft
    "__exclude_sync_diff_attrs__",  # syft
    "__repr_attrs__",  # syft
    "get_sync_dependencies",
    "_data_repr",
    "syft_eq",  # syft
    "__table_coll_widths__",
    "_clear_cache",
    "_set_reprs",
    "get_api",
    "get_api_wrapped",
]
dont_wrap_output_attrs = [
    "id",
    "refresh_object",
    "__repr__",
    "__str__",
    "__repr_attrs__",
    "_repr_html_",
    "_repr_markdown_",
    "_repr_latex_",
    "__array_struct__",
    "__array_prepare__",
    "__array_wrap__",
    "__bool__",
    "__len__",
    "syft_resolved",  # syft
    "server_uid",
    "syft_action_data_server_id",
    "__sha256__",
    "__hash_exclude_attrs__",
    "__exclude_sync_diff_attrs__",  # syft
    "__repr_attrs__",  # syft
    "get_sync_dependencies",  # syft
    "syft_eq",  # syft
    "__table_coll_widths__",
    "_clear_cache",
    "_set_reprs",
    "get_api",
    "get_api_wrapped",
]
dont_make_side_effects = [
    "refresh_object",
    "id",
    "__repr_attrs__",
    "_repr_html_",
    "_repr_markdown_",
    "_repr_latex_",
    "__repr__",
    "__getitem__",
    "__setitem__",
    "__len__",
    "shape",
    "syft_resolved",  # syft
    "server_uid",
    "syft_action_data_server_id",
    "__sha256__",
    "__hash_exclude_attrs__",
    "__exclude_sync_diff_attrs__",  # syft
    "__repr_attrs__",
    "get_sync_dependencies",
    "syft_eq",  # syft
    "__table_coll_widths__",
    "_clear_cache",
    "_set_reprs",
    "get_api",
    "get_api_wrapped",
]
action_data_empty_must_run = [
    "__repr__",
    "__str__",
]

methods_to_check_in_cache = [
    "_ipython_display_",
    "_repr_mimebundle_",
    "_repr_latex_",
    "_repr_javascript_",
    "_repr_html_",
    "_repr_jpeg_",
    "_repr_png_",
    "_repr_svg_",
    "_repr_pretty_",
    "_repr_pdf_",
    "_repr_json_",
    "_repr_markdown_",
]


class PreHookContext(SyftBaseObject):
    __canonical_name__ = "PreHookContext"
    __version__ = SYFT_OBJECT_VERSION_1

    """Hook context

    Parameters:
        obj: Any
            The ActionObject to use for the action
        op_name: str
            The method name to use for the action
        server_uid: Optional[UID]
            Optional Syft server UID
        result_id: Optional[Union[UID, LineageID]]
            Optional result Syft UID
        action: Optional[Action]
            The action generated by the current hook
    """

    obj: Any = None
    op_name: str
    server_uid: UID | None = None
    result_id: UID | LineageID | None = None
    result_twin_type: TwinMode | None = None
    action: Action | None = None
    action_type: ActionType | None = None


@as_result(SyftException)
def make_action_side_effect(
    context: PreHookContext, *args: Any, **kwargs: Any
) -> tuple[PreHookContext, tuple[Any, ...], dict[str, Any]]:
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
    try:
        action = context.obj.syft_make_action_with_self(
            op=context.op_name,
            args=args,
            kwargs=kwargs,
            action_type=context.action_type,
        )
        context.action = action
    except Exception as e:
        raise SyftException(public_message=f"make_action_side_effect failed {e}") from e

    return context, args, kwargs


class TraceResultRegistry:
    __result_registry__: dict[int, TraceResult] = {}

    @classmethod
    def set_trace_result_for_current_thread(
        cls,
        client: SyftClient,
    ) -> None:
        cls.__result_registry__[threading.get_ident()] = TraceResult(
            client=client, is_tracing=True
        )

    @classmethod
    def get_trace_result_for_thread(cls) -> TraceResult | None:
        return cls.__result_registry__.get(threading.get_ident(), None)

    @classmethod
    def reset_result_for_thread(cls) -> None:
        if threading.get_ident() in cls.__result_registry__:
            del cls.__result_registry__[threading.get_ident()]

    @classmethod
    def current_thread_is_tracing(cls) -> bool:
        trace_result = cls.get_trace_result_for_thread()
        if trace_result is None:
            return False
        else:
            return trace_result.is_tracing


class TraceResult(SyftBaseModel):
    result: list = []
    client: SyftClient
    is_tracing: bool = False


@as_result(SyftException)
def trace_action_side_effect(
    context: PreHookContext, *args: Any, **kwargs: Any
) -> tuple[PreHookContext, tuple[Any, ...], dict[str, Any]]:
    action = context.action
    if action is not None and TraceResultRegistry.current_thread_is_tracing():
        trace_result = TraceResultRegistry.get_trace_result_for_thread()
        trace_result.result += [action]  # type: ignore
    return context, args, kwargs


def convert_to_pointers(
    api: SyftAPI,
    server_uid: UID | None = None,
    args: list | None = None,
    kwargs: dict | None = None,
) -> tuple[list, dict]:
    # relative
    from ..dataset.dataset import Asset

    def process_arg(arg: ActionObject | Asset | UID | Any) -> Any:
        if (
            not isinstance(arg, ActionObject | Asset | UID)
            and api.signing_key is not None  # type: ignore[unreachable]
        ):
            arg = ActionObject.from_obj(  # type: ignore[unreachable]
                syft_action_data=arg,
                syft_client_verify_key=api.signing_key.verify_key,
                syft_server_location=api.server_uid,
            )
            arg.syft_server_uid = server_uid
            r = arg._save_to_blob_storage().unwrap()
            if isinstance(r, SyftWarning):
                logger.debug(r.message)
            arg = api.services.action.set(arg)
        return arg

    arg_list = [process_arg(arg) for arg in args] if args else []
    kwarg_dict = {k: process_arg(v) for k, v in kwargs.items()} if kwargs else {}

    return arg_list, kwarg_dict


@as_result(SyftException)
def send_action_side_effect(
    context: PreHookContext, *args: Any, **kwargs: Any
) -> tuple[PreHookContext, tuple[Any, ...], dict[str, Any]]:
    """Create a new action from the context.op_name, and execute it on the remote server."""
    try:
        if context.action is None:
            context, _, _ = make_action_side_effect(context, *args, **kwargs).unwrap()

        action_result = context.obj.syft_execute_action(context.action, sync=True)

        if not isinstance(action_result, ActionObject):
            raise SyftException(
                public_message=f"Got back unexpected response : {action_result}"
            )
        else:
            context.server_uid = action_result.syft_server_uid
            context.result_id = action_result.id
            context.result_twin_type = action_result.syft_twin_type
    except Exception as e:
        # print(e)
        raise SyftException(public_message=f"send_action_side_effect failed {e}") from e
    return context, args, kwargs


@as_result(SyftException)
def propagate_server_uid(context: PreHookContext, op: str, result: Any) -> Any:
    """Patch the result to include the syft_server_uid

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
        context.obj, "syft_server_uid"
    ):
        return result

    try:
        syft_server_uid = getattr(context.obj, "syft_server_uid", None)
        if syft_server_uid is None:
            raise SyftException(
                public_message="Can't proagate server_uid because parent doesnt have one"
            )

        if op not in context.obj._syft_dont_wrap_attrs():
            if hasattr(result, "syft_server_uid"):
                result.syft_server_uid = syft_server_uid
        else:
            raise SyftException(
                public_message="dont propogate server_uid because output isnt wrapped"
            )
    except Exception:
        raise SyftException(public_message="propagate_server_uid failed")

    return result


def debox_args_and_kwargs(args: Any, kwargs: Any) -> tuple[Any, Any]:
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


BASE_PASSTHROUGH_ATTRS: list[str] = [
    "id",
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
    "is_link",
    "wait",
    "_save_to_blob_storage",
    "_save_to_blob_storage_",
    "syft_action_data",
    "__check_action_data",
    "as_empty_data",
    "_set_obj_location_",
    "syft_action_data_cache",
    "reload_cache",
    "syft_resolved",
    "refresh_object",
    "syft_action_data_server_id",
    "server_uid",
    "__sha256__",
    "__hash_exclude_attrs__",
    "__hash__",
    "create_shareable_sync_copy",
    "_has_private_sync_attrs",
    "__exclude_sync_diff_attrs__",
    "__repr_attrs__",
    "get_sync_dependencies",
    "_data_repr",
    "syft_eq",
    "__table_coll_widths__",
    "_clear_cache",
    "_set_reprs",
    "get_api",
    "get_api_wrapped",
]


def truncate_str(string: str, length: int = 100) -> str:
    stringlen = len(string)
    if stringlen > length:
        n_hidden = stringlen - length
        string = f"{string[:length]}... ({n_hidden} characters hidden)"
    return string


@serializable(without=["syft_pre_hooks__", "syft_post_hooks__"])
class ActionObject(SyncableSyftObject):
    """Action object for remote execution."""

    __canonical_name__ = "ActionObject"
    __version__ = SYFT_OBJECT_VERSION_1
    __private_sync_attr_mocks__: ClassVar[dict[str, Any]] = {
        "syft_action_data_cache": None,
        "syft_blob_storage_entry_id": None,
    }

    __attr_searchable__: list[str] = []  # type: ignore[misc]
    syft_action_data_cache: Any | None = None
    syft_blob_storage_entry_id: UID | None = None
    syft_pointer_type: ClassVar[type[ActionObjectPointer]]

    # Help with calculating history hash for code verification
    syft_parent_hashes: int | list[int] | None = None
    syft_parent_op: str | None = None
    syft_parent_args: Any | None = None
    syft_parent_kwargs: Any | None = None
    syft_history_hash: int | None = None
    syft_internal_type: ClassVar[type[Any]]
    syft_server_uid: UID | None = None
    syft_pre_hooks__: dict[str, list] = {}
    syft_post_hooks__: dict[str, list] = {}
    syft_twin_type: TwinMode = TwinMode.NONE
    syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS
    syft_action_data_type: type | None = None
    syft_action_data_repr_: str | None = None
    syft_action_data_str_: str | None = None
    syft_has_bool_attr: bool | None = None
    syft_resolve_data: bool | None = None
    syft_created_at: DateTime | None = None
    syft_resolved: bool = True
    syft_action_data_server_id: UID | None = None
    syft_action_saved_to_blob_store: bool = True
    # syft_dont_wrap_attrs = ["shape"]

    def syft_get_diffs(self, ext_obj: Any) -> list[AttrDiff]:
        # relative
        from ...service.sync.diff_state import AttrDiff

        diff_attrs = []

        # Sanity check
        if ext_obj.id != self.id:
            raise Exception("Not the same id for low side and high side requests")

        low_data = ext_obj.syft_action_data
        high_data = self.syft_action_data

        try:
            cmp = low_data != high_data
            if isinstance(cmp, Iterable):
                cmp = all(cmp)
        except Exception:
            cmp = False

        if cmp:
            diff_attr = AttrDiff(
                attr_name="syft_action_data", low_attr=low_data, high_attr=high_data
            )
            diff_attrs.append(diff_attr)
        return diff_attrs

    def _set_obj_location_(self, server_uid: UID, credentials: SyftVerifyKey) -> None:
        self.syft_server_location = server_uid
        self.syft_client_verify_key = credentials
        if self.syft_action_data_server_id is None:
            self.syft_action_data_server_id = server_uid

    @property
    def syft_action_data(self) -> Any:
        if self.syft_blob_storage_entry_id and self.syft_created_at:
            self.reload_cache()
        return self.syft_action_data_cache

    def reload_cache(self) -> None:
        # If ActionDataEmpty then try to fetch it from store.
        if isinstance(self.syft_action_data_cache, ActionDataEmpty):
            blob_storage_read_method = from_api_or_context(
                func_or_path="blob_storage.read",
                syft_server_location=self.syft_server_location,
                syft_client_verify_key=self.syft_client_verify_key,
            )

            if blob_storage_read_method is not None:
                blob_retrieval_object = blob_storage_read_method(
                    uid=self.syft_blob_storage_entry_id
                )

                # relative
                from ...store.blob_storage import BlobRetrieval

                if isinstance(blob_retrieval_object, BlobRetrieval):
                    # TODO: This change is temporary to for gateway to be compatible with the new blob storage
                    self.syft_action_data_cache = blob_retrieval_object.read()
                    self.syft_action_data_type = type(self.syft_action_data_cache)
                    return None
                else:
                    # In the case of gateway, we directly receive the actual object
                    # TODO: The ideal solution would be to stream the object from the datasite through the gateway
                    # Currently , we are just passing the object as it is, which would be fixed later.
                    self.syft_action_data_cache = blob_retrieval_object
                    self.syft_action_data_type = type(self.syft_action_data_cache)
                    return None
            else:
                raise SyftException(
                    public_meesage="Could not reload cache, could not get read method"
                )

        return None

    def _save_to_blob_storage_(self, data: Any) -> SyftWarning | None:
        # relative
        from ...types.blob_storage import BlobFile
        from ...types.blob_storage import CreateBlobStorageEntry

        if not isinstance(data, ActionDataEmpty):
            if isinstance(data, BlobFile):
                if not data.uploaded:
                    api = self.get_api()
                    data._upload_to_blobstorage_from_api(api)
            else:
                get_metadata = from_api_or_context(
                    func_or_path="metadata.get_metadata",
                    syft_server_location=self.syft_server_location,
                    syft_client_verify_key=self.syft_client_verify_key,
                )
                if (
                    get_metadata is not None
                    and not can_upload_to_blob_storage(data, get_metadata()).unwrap()
                ):
                    self.syft_action_saved_to_blob_store = False
                    return SyftWarning(
                        message=(
                            f"The action object {self.id} was not saved to"
                            f" the blob store but to memory cache since it is small."
                        )
                    )
                serialized = serialize(data, to_bytes=True)
                size = sys.getsizeof(serialized)
                storage_entry = CreateBlobStorageEntry.from_obj(data, file_size=size)

                if not TraceResultRegistry.current_thread_is_tracing():
                    self.syft_action_data_cache = self.as_empty_data()
                if self.syft_blob_storage_entry_id is not None:
                    # TODO: check if it already exists
                    storage_entry.id = self.syft_blob_storage_entry_id
                allocate_method = from_api_or_context(
                    func_or_path="blob_storage.allocate",
                    syft_server_location=self.syft_server_location,
                    syft_client_verify_key=self.syft_client_verify_key,
                )
                if allocate_method is not None:
                    blob_deposit_object = allocate_method(storage_entry)
                    blob_deposit_object.write(BytesIO(serialized)).unwrap()
                    self.syft_blob_storage_entry_id = (
                        blob_deposit_object.blob_storage_entry_id
                    )
                else:
                    logger.warn("cannot save to blob storage. allocate_method=None")

            self.syft_action_data_type = type(data)
            self._set_reprs(data)
            self.syft_has_bool_attr = hasattr(data, "__bool__")
        else:
            logger.debug(
                "skipping writing action object to store, passed data was empty."
            )

        self.syft_action_data_cache = data

        return None

    @as_result(SyftException)
    def _save_to_blob_storage(
        self, allow_empty: bool = False
    ) -> SyftSuccess | SyftWarning:
        data = self.syft_action_data

        if isinstance(data, ActionDataEmpty):
            raise SyftException(
                public_message=f"cannot store empty object {self.id} to the blob storage"
            )

        result = self._save_to_blob_storage_(data)
        if isinstance(result, SyftWarning):
            return result
        if not TraceResultRegistry.current_thread_is_tracing():
            self._clear_cache()
        return SyftSuccess(message=f"Saved action object {self.id} to the blob store")

    def _clear_cache(self) -> None:
        self.syft_action_data_cache = self.as_empty_data()

    def _set_reprs(self, data: any) -> None:
        if inspect.isclass(data):
            self.syft_action_data_repr_ = truncate_str(repr_cls(data))
        else:
            self.syft_action_data_repr_ = truncate_str(
                data._repr_markdown_()
                if hasattr(data, "_repr_markdown_")
                else data.__repr__()
            )
        self.syft_action_data_str_ = truncate_str(str(data))

    @property
    def is_pointer(self) -> bool:
        return self.syft_server_uid is not None

    @property
    def syft_lineage_id(self) -> LineageID:
        """Compute the LineageID of the ActionObject, using the `id` and the `syft_history_hash` members"""
        return LineageID(self.id, self.syft_history_hash)

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def __check_action_data(cls, values: Any) -> dict:
        if isinstance(values, dict):
            v = values.get("syft_action_data_cache")
            if values.get("syft_action_data_type", None) is None:
                values["syft_action_data_type"] = type(v)
            if not isinstance(v, ActionDataEmpty):
                if inspect.isclass(v):
                    values["syft_action_data_repr_"] = truncate_str(repr_cls(v))
                else:
                    values["syft_action_data_repr_"] = truncate_str(
                        v._repr_markdown_()
                        if v is not None and hasattr(v, "_repr_markdown_")
                        else v.__repr__()
                    )
                values["syft_action_data_str_"] = truncate_str(str(v))
                values["syft_has_bool_attr"] = hasattr(v, "__bool__")

        return values

    @property
    def is_mock(self) -> bool:
        return self.syft_twin_type == TwinMode.MOCK

    @property
    def is_real(self) -> bool:
        return self.syft_twin_type == TwinMode.PRIVATE

    @property
    def is_twin(self) -> bool:
        return self.syft_twin_type != TwinMode.NONE

    def syft_point_to(self, server_uid: UID) -> ActionObject:
        """Set the syft_server_uid, used in the post hooks"""
        self.syft_server_uid = server_uid
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

    def syft_eq(self, ext_obj: Self | None) -> bool:
        if ext_obj is None:
            return False
        return self.id.id == ext_obj.id.id

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
        if self.syft_server_uid is None:
            raise SyftException(
                public_message="Pointers can't execute without a server_uid."
            )

        # relative
        from ...client.api import SyftAPICall

        api = self.get_api()
        # if api is None:
        #     raise ValueError(f"api is None. You must login to {self.syft_server_uid}")
        kwargs = {"action": action}
        api_call = SyftAPICall(
            server_uid=self.syft_server_uid,
            path="action.execute",
            args=[],
            kwargs=kwargs,
        )
        res = api.make_call(api_call)
        if isinstance(res, SyftSuccess):
            return res.value  # type: ignore[return-value]
        return res  # type: ignore[return-value]

    def request(self, client: SyftClient) -> Any:
        # relative
        from ..request.request import ActionStoreChange
        from ..request.request import SubmitRequest

        action_object_link = LinkedObject.from_obj(
            self, server_uid=self.syft_server_uid
        )
        permission_change = ActionStoreChange(
            linked_obj=action_object_link, apply_permission_type=ActionPermission.READ
        )
        if client.credentials is None:
            raise SyftException(public_message=f"{client} has no signing key")
        submit_request = SubmitRequest(
            changes=[permission_change],
            requesting_user_verify_key=client.credentials.verify_key,
        )
        return client.api.services.request.submit(submit_request)

    def _syft_try_to_save_to_store(self, obj: SyftObject) -> None:
        if self.syft_server_uid is None or self.syft_client_verify_key is None:
            return
        elif obj.syft_server_uid is not None:
            return

        if obj.syft_blob_storage_entry_id is not None:
            return
        # TODO fix: the APIRegistry often gets the wrong client
        # if you have 2 clients in memory
        # therefore the following happens if you call a method
        # with a pointer to a twin (mock version)
        # 1) it gets the wrong credentials
        # 2) it uses the mock version to overwrite the real version
        # 3) it shouldnt send in the first place as it already exists

        # relative

        if obj.syft_server_location is None:
            obj.syft_server_location = obj.syft_server_uid

        action = Action(
            path="",
            op="",
            remote_self=None,
            result_id=obj.id,
            args=[],
            kwargs={},
            action_type=ActionType.CREATEOBJECT,
            create_object=obj,
        )

        if TraceResultRegistry.current_thread_is_tracing():
            trace_result = TraceResultRegistry.get_trace_result_for_thread()
            trace_result.result += [action]  # type: ignore

        api = self.get_api_wrapped().unwrap(
            public_message=(
                f"Failed saving {obj} to blob storage, api is None."
                f" You must login to {self.syft_server_location}."
            )
        )

        obj._set_obj_location_(api.server_uid, api.signing_key.verify_key)  # type: ignore[union-attr]

        try:
            api.services.action.execute(action)
        except Exception as e:
            print(f"Failed to to store (arg) {obj} to store, {e}")

    def _syft_prepare_obj_uid(self, obj: Any) -> LineageID:
        # We got the UID
        if isinstance(obj, UID | LineageID):
            return LineageID(obj.id)

        # We got the ActionObjectPointer
        if isinstance(obj, ActionObjectPointer):
            return obj.syft_lineage_id

        # We got the ActionObject. We need to save it in the store.
        if isinstance(obj, ActionObject):
            self._syft_try_to_save_to_store(obj)
            return obj.syft_lineage_id

        # We got a raw object. We need to create the ActionObject from scratch and save it in the store.
        act_obj = ActionObject.from_obj(
            obj,
            syft_client_verify_key=self.syft_client_verify_key,
            syft_server_location=self.syft_server_location,
        )

        self._syft_try_to_save_to_store(act_obj)

        return act_obj.syft_lineage_id

    def syft_make_action(
        self,
        path: str,
        op: str,
        remote_self: UID | LineageID | None = None,
        args: (
            list[UID | LineageID | ActionObjectPointer | ActionObject | Any] | None
        ) = None,
        kwargs: (
            dict[str, UID | LineageID | ActionObjectPointer | ActionObject | Any] | None
        ) = None,
        action_type: ActionType | None = None,
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

        arg_ids = [self._syft_prepare_obj_uid(obj) for obj in args]

        kwarg_ids = {k: self._syft_prepare_obj_uid(obj) for k, obj in kwargs.items()}

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
        args: list[UID | ActionObjectPointer] | None = None,
        kwargs: dict[str, UID | ActionObjectPointer] | None = None,
        action_type: ActionType | None = None,
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

    def get_sync_dependencies(
        self, context: AuthedServiceContext, **kwargs: dict
    ) -> list[UID]:
        try:
            job = context.server.services.job.get_by_result_id(context, self.id.id)
            return [job.id]
        except SyftException:
            return []

    def syft_get_path(self) -> str:
        """Get the type path of the underlying object"""
        if (
            isinstance(self, AnyActionObject)
            and self.syft_internal_type
            and self.syft_action_data_type is not None
        ):
            # avoids AnyActionObject errors
            return f"{self.syft_action_data_type.__name__}"
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
            *args: list[UID | ActionObjectPointer] | None,
            **kwargs: dict[str, UID | ActionObjectPointer] | None,
        ) -> Action:
            return self.syft_make_action_with_self(op=op, args=args, kwargs=kwargs)

        return wrapper

    def send(self, client: SyftClient) -> Any:
        return self._send(
            server_uid=client.id,
            verify_key=client.verify_key,
            add_storage_permission=True,
        )

    def _send(
        self,
        server_uid: UID,
        verify_key: SyftVerifyKey,
        add_storage_permission: bool = True,
    ) -> Self:
        self._set_obj_location_(server_uid, verify_key)

        blob_storage_res = self._save_to_blob_storage().unwrap()
        api = self._get_api()

        if isinstance(blob_storage_res, SyftWarning):
            logger.debug(blob_storage_res.message)
        res = api.services.action.set(
            self,
            add_storage_permission=add_storage_permission,
        )
        if isinstance(res, ActionObject):
            self.syft_created_at = res.syft_created_at
        return res

    def get_from(self, client: SyftClient) -> Any:
        """Get the object from a Syft Client"""
        res = client.api.services.action.get(self.id)
        return res.syft_action_data

    def refresh_object(self, resolve_nested: bool = True) -> ActionObject:
        # relative
        return self.get_api().services.action.get(
            self.id, resolve_nested=resolve_nested
        )

    def has_storage_permission(self) -> bool:
        try:
            api = self.get_api()
            return api.services.action.has_storage_permission(self.id.id)
        except Exception:
            return False

    def get(self, block: bool = False) -> Any:
        """Get the object from a Syft Client"""
        # relative

        if block:
            self.wait()

        res = self.refresh_object()
        if not isinstance(res, ActionObject):
            raise SyftException(public_message=f"{res}")  # type: ignore
        elif issubclass(res.syft_action_data_type, Err):
            raise SyftException(public_message=f"{res.syft_action_data.err()}")
        else:
            if not self.has_storage_permission():
                prompt_warning_message(
                    message="This is a placeholder object, the real data lives on a different server and is not synced."
                )
            nested_res = res.syft_action_data
            if isinstance(nested_res, ActionObject):
                nested_res.syft_server_location = res.syft_server_location
                nested_res.syft_client_verify_key = res.syft_client_verify_key
            return nested_res

    def as_empty(self) -> ActionObject:
        id = self.id

        if isinstance(id, LineageID):
            id = id.id

        res = ActionObject.empty(
            self.syft_internal_type,
            id,
            self.syft_lineage_id,
            self.syft_resolved,
            syft_blob_storage_entry_id=self.syft_blob_storage_entry_id,
        )
        if isinstance(self.id, LineageID):
            res.id = self.id

        return res

    def create_shareable_sync_copy(self, mock: bool) -> ActionObject:
        if mock:
            res = self.as_empty()
            for k, v in self.__private_sync_attr_mocks__.items():
                setattr(res, k, v)
            return res
        return self

    @staticmethod
    def from_path(
        path: str | Path,
        id: UID | None = None,
        syft_lineage_id: LineageID | None = None,
        syft_client_verify_key: SyftVerifyKey | None = None,
        syft_server_location: UID | None = None,
    ) -> ActionObject:
        """Create an Action Object from a file."""
        # relative
        from ...types.blob_storage import BlobFile

        if id is not None and syft_lineage_id is not None and id != syft_lineage_id.id:
            raise ValueError("UID and LineageID should match")

        _path = path if isinstance(path, Path) else Path(path)
        syft_action_data = BlobFile(path=_path, file_name=_path.name)

        action_type = action_type_for_object(syft_action_data)

        action_object = action_type(syft_action_data_cache=syft_action_data)

        if id is not None:
            action_object.id = id

        if syft_client_verify_key is not None:
            action_object.syft_client_verify_key = syft_client_verify_key

        if syft_server_location is not None:
            action_object.syft_server_location = syft_server_location

        if syft_lineage_id is not None:
            action_object.id = syft_lineage_id.id
            action_object.syft_history_hash = syft_lineage_id.syft_history_hash
        elif id is not None:
            action_object.syft_history_hash = hash(id)

        return action_object

    @staticmethod
    def from_obj(
        syft_action_data: Any,
        id: UID | None = None,
        syft_lineage_id: LineageID | None = None,
        syft_client_verify_key: SyftVerifyKey | None = None,
        syft_server_location: UID | None = None,
        syft_resolved: bool | None = True,
        data_server_id: UID | None = None,
        syft_blob_storage_entry_id: UID | None = None,
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
        if id is not None and syft_lineage_id is not None and id != syft_lineage_id.id:
            raise ValueError("UID and LineageID should match")

        # check if the object's type is supported
        try:
            SyftObjectRegistry.get_canonical_name_version(syft_action_data)
        except Exception:
            obj_type = type(syft_action_data)
            raise SyftException(
                public_message=(
                    f"Error when creating action object for {syft_action_data}.\n"
                    f"Unsupported data type: '{obj_type.__module__}.{obj_type.__name__}'"
                )
            )

        action_type = action_type_for_object(syft_action_data)
        action_object = action_type(syft_action_data_cache=syft_action_data)
        action_object.syft_blob_storage_entry_id = syft_blob_storage_entry_id
        action_object.syft_action_data_server_id = data_server_id
        action_object.syft_resolved = syft_resolved

        if id is not None:
            action_object.id = id

        if syft_client_verify_key is not None:
            action_object.syft_client_verify_key = syft_client_verify_key

        if syft_server_location is not None:
            action_object.syft_server_location = syft_server_location

        if syft_lineage_id is not None:
            action_object.id = syft_lineage_id.id
            action_object.syft_history_hash = syft_lineage_id.syft_history_hash
        elif id is not None:
            action_object.syft_history_hash = hash(id)

        return action_object

    @classmethod
    def add_trace_hook(cls) -> bool:
        return True
        # if trace_action_side_effect not in self.syft_pre_hooks__[HOOK_ALWAYS]:
        #     self.syft_pre_hooks__[HOOK_ALWAYS].append(trace_action_side_effect)

    @classmethod
    def remove_trace_hook(cls) -> bool:
        return True
        # self.syft_pre_hooks__[HOOK_ALWAYS].pop(trace_action_side_effct, None)

    def as_empty_data(self) -> ActionDataEmpty:
        return ActionDataEmpty(syft_internal_type=self.syft_internal_type)

    def wait(self, timeout: int | None = None) -> ActionObject:
        # relative

        api = self.get_api()
        if isinstance(self.id, LineageID):
            obj_id = self.id.id
        else:
            obj_id = self.id

        counter = 0
        while api:
            obj_resolved: bool | str = api.services.action.is_resolved(obj_id)
            if isinstance(obj_resolved, str):
                raise SyftException(public_message=obj_resolved)
            if obj_resolved:
                break
            if not obj_resolved:
                time.sleep(1)
                if timeout is not None:
                    counter += 1
                    if counter > timeout:
                        raise SyftException(public_message="Reached Timeout!")
        return self

    @staticmethod
    def link(
        result_id: UID,
        pointer_id: UID | None = None,
    ) -> ActionObject:
        link = ActionDataLink(action_object_id=pointer_id)
        res = ActionObject.from_obj(
            id=result_id,
            syft_action_data=link,
        )
        return res

    @staticmethod
    def obj_not_ready(
        id: UID, syft_server_location: UID, syft_client_verify_key: SyftVerifyKey
    ) -> ActionObject:
        inner_obj = ObjectNotReady(obj_id=id)

        res = ActionObject.from_obj(
            id=id,
            syft_action_data=inner_obj,
            syft_resolved=False,
            syft_server_location=syft_server_location,
            syft_client_verify_key=syft_client_verify_key,
        )
        return res

    @classmethod
    def empty(
        # TODO: fix the mypy issue
        cls,
        syft_internal_type: type[Any] | None = None,
        id: UID | None = None,
        syft_lineage_id: LineageID | None = None,
        syft_resolved: bool | None = True,
        data_server_id: UID | None = None,
        syft_blob_storage_entry_id: UID | None = None,
    ) -> Self:
        """Create an ActionObject from a type, using a ActionDataEmpty object

        Parameters:
            syft_internal_type: Type
                The Type for which to create a ActionDataEmpty object
            id: Optional[UID]
                Which ID to use for the ActionObject. Optional
            syft_lineage_id: Optional[LineageID]
                Which LineageID to use for the ActionObject. Optional
        """

        syft_internal_type = (
            type(None) if syft_internal_type is None else syft_internal_type
        )
        empty = ActionDataEmpty(syft_internal_type=syft_internal_type)
        res = cls.from_obj(
            id=id,
            syft_lineage_id=syft_lineage_id,
            syft_action_data=empty,
            syft_resolved=syft_resolved,
            data_server_id=data_server_id,
            syft_blob_storage_entry_id=syft_blob_storage_entry_id,
        )
        res.__dict__["syft_internal_type"] = syft_internal_type
        return res

    def __post_init__(self) -> None:
        """Add pre/post hooks."""
        if HOOK_ALWAYS not in self.syft_pre_hooks__:
            self.syft_pre_hooks__[HOOK_ALWAYS] = []

        if HOOK_ON_POINTERS not in self.syft_pre_hooks__:
            self.syft_pre_hooks__[HOOK_ON_POINTERS] = []

        if HOOK_ALWAYS not in self.syft_post_hooks__:
            self.syft_post_hooks__[HOOK_ALWAYS] = []

        if HOOK_ON_POINTERS not in self.syft_post_hooks__:
            self.syft_post_hooks__[HOOK_ON_POINTERS] = []

        api = self.get_api_wrapped()

        eager_execution_enabled = (
            api.is_ok()
            and api.unwrap().metadata is not None
            and api.unwrap().metadata.eager_execution_enabled  # type: ignore
        )

        self._syft_add_pre_hooks__(eager_execution_enabled)
        self._syft_add_post_hooks__(eager_execution_enabled)

        if isinstance(self.syft_action_data_type, ActionObject):
            raise Exception("Nested ActionObjects", self.syft_action_data_repr_)

        self.syft_history_hash = hash(self.id)

    def _syft_add_pre_hooks__(self, eager_execution: bool) -> None:
        """
        Add pre-hooks

        Args:
            eager_execution: bool: If eager execution is enabled, hooks for
                tracing and executing the action on remote are added.
        """

        # this should be a list as orders matters
        for side_effect in [make_action_side_effect]:
            if side_effect not in self.syft_pre_hooks__[HOOK_ALWAYS]:
                self.syft_pre_hooks__[HOOK_ALWAYS].append(side_effect)

        if eager_execution:
            for side_effect in [send_action_side_effect]:
                if side_effect not in self.syft_pre_hooks__[HOOK_ON_POINTERS]:
                    self.syft_pre_hooks__[HOOK_ON_POINTERS].append(side_effect)

            if trace_action_side_effect not in self.syft_pre_hooks__[HOOK_ALWAYS]:
                self.syft_pre_hooks__[HOOK_ALWAYS].append(trace_action_side_effect)

    def _syft_add_post_hooks__(self, eager_execution: bool) -> None:
        """
        Add post-hooks

        Args:
            eager_execution: bool: If eager execution is enabled, hooks for
                tracing and executing the action on remote are added.
        """
        if eager_execution:
            # this should be a list as orders matters
            for side_effect in [propagate_server_uid]:
                if side_effect not in self.syft_post_hooks__[HOOK_ALWAYS]:
                    self.syft_post_hooks__[HOOK_ALWAYS].append(side_effect)

    def _syft_run_pre_hooks__(
        self, context: PreHookContext, name: str, args: Any, kwargs: Any
    ) -> tuple[PreHookContext, tuple[Any, ...], dict[str, Any]]:
        """Hooks executed before the actual call"""
        result_args, result_kwargs = args, kwargs
        if name in self.syft_pre_hooks__:
            for hook in self.syft_pre_hooks__[name]:
                result = hook(context, *result_args, **result_kwargs)
                if result.is_ok():
                    context, result_args, result_kwargs = result.ok()
                else:
                    logger.debug(f"Pre-hook failed with {result.err()}")
        if name not in self._syft_dont_wrap_attrs():
            if HOOK_ALWAYS in self.syft_pre_hooks__:
                for hook in self.syft_pre_hooks__[HOOK_ALWAYS]:
                    result = hook(context, *result_args, **result_kwargs)
                    if result.is_ok():
                        context, result_args, result_kwargs = result.ok()
                    else:
                        msg = str(result.err())
                        msg = msg.replace("\\n", "\n")
                        logger.debug(f"Pre-hook failed with {msg}")

        if self.is_pointer:
            if name not in self._syft_dont_wrap_attrs():
                if HOOK_ALWAYS in self.syft_pre_hooks__:
                    for hook in self.syft_pre_hooks__[HOOK_ON_POINTERS]:
                        result = hook(context, *result_args, **result_kwargs)
                        if result.is_ok():
                            context, result_args, result_kwargs = result.ok()
                        else:
                            msg = str(result.err())
                            msg = msg.replace("\\n", "\n")
                            logger.debug(f"Pre-hook failed with {msg}")

        return context, result_args, result_kwargs

    def _syft_run_post_hooks__(
        self, context: PreHookContext, name: str, result: Any
    ) -> Any:
        """Hooks executed after the actual call"""
        new_result = result
        if name in self.syft_post_hooks__:
            for hook in self.syft_post_hooks__[name]:
                result = hook(context, name, new_result)
                if result.is_ok():
                    new_result = result.ok()
                else:
                    logger.debug(f"Post hook failed with {result.err()}")

        if name not in self._syft_dont_wrap_attrs():
            if HOOK_ALWAYS in self.syft_post_hooks__:
                for hook in self.syft_post_hooks__[HOOK_ALWAYS]:
                    result = hook(context, name, new_result)
                    if result.is_ok():
                        new_result = result.ok()
                    else:
                        logger.debug(f"Post hook failed with {result.err()}")

        if self.is_pointer:
            if name not in self._syft_dont_wrap_attrs():
                if HOOK_ALWAYS in self.syft_post_hooks__:
                    for hook in self.syft_post_hooks__[HOOK_ON_POINTERS]:
                        result = hook(context, name, new_result)
                        if result.is_ok():
                            new_result = result.ok()
                        else:
                            logger.debug(f"Post hook failed with {result.err()}")

        return new_result

    def _syft_output_action_object(
        self, result: Any, context: PreHookContext | None = None
    ) -> Any:
        """Wrap the result in an ActionObject"""
        if issubclass(type(result), ActionObject):
            return result

        constructor = action_type_for_type(result)
        syft_twin_type = TwinMode.NONE
        if context is not None and context.result_twin_type is not None:
            syft_twin_type = context.result_twin_type
        result = constructor(
            syft_twin_type=syft_twin_type,
            syft_action_data_cache=result,
            syft_server_location=self.syft_server_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        return result

    def _syft_passthrough_attrs(self) -> list[str]:
        """These attributes are forwarded to the `object` base class."""
        return passthrough_attrs + getattr(self, "syft_passthrough_attrs", [])

    def _syft_dont_wrap_attrs(self) -> list[str]:
        """The results from these attributes are ignored from UID patching."""
        return dont_wrap_output_attrs + getattr(self, "syft_dont_wrap_attrs", [])

    def _syft_get_attr_context(self, name: str) -> Any:
        """Find which instance - Syft ActionObject or the original object - has the requested attribute."""
        defined_on_self = name in self.__dict__ or name in self.__private_attributes__

        # use the custom defined version
        context_self = self
        if not defined_on_self:
            context_self = self.syft_action_data

        return context_self

    def _syft_attr_propagate_ids(
        self, context: PreHookContext, name: str, result: Any
    ) -> Any:
        """Patch the results with the syft_history_hash, server_uid, and result_id."""
        if name in self._syft_dont_wrap_attrs():
            return result

        # Wrap as Syft Object
        result = self._syft_output_action_object(result, context)

        # Propagate History
        if context.action is not None:
            result.syft_history_hash = context.action.syft_history_hash

        # Propagate Syft Server UID
        result.syft_server_uid = context.server_uid

        # Propogate Syft Server Location and Client Verify Key
        result.syft_server_location = context.syft_server_location
        result.syft_client_verify_key = context.syft_client_verify_key

        # Propagate Syft blob storage entry id
        object_attrs = [
            "syft_blob_storage_entry_id",
            "syft_action_data_repr_",
            "syft_action_data_str_",
            "syft_action_data_type",
        ]
        for attr_name in object_attrs:
            attr_value = getattr(context.obj, attr_name, None)
            setattr(result, attr_name, attr_value)

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

        if self.syft_has_bool_attr:
            raise RuntimeError(
                "[_wrap_attribute_for_bool_on_nonbools] self.syft_action_data already implements the bool operator"
            )

        logger.debug("[__getattribute__] Handling bool on nonbools")
        context = PreHookContext(
            obj=self,
            op_name=name,
            syft_server_location=self.syft_server_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        context, _, _ = self._syft_run_pre_hooks__(context, name, (), {})

        # no input needs to propagate
        result = self._syft_run_post_hooks__(
            context,
            name,
            any(
                x is not None
                for x in (self.syft_blob_storage_entry_id, self.syft_action_data_cache)
            ),
        )
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
        logger.debug(f"[__getattribute__] Handling property {name}")

        context = PreHookContext(
            obj=self,
            op_name=name,
            action_type=ActionType.GETATTRIBUTE,
            syft_server_location=self.syft_server_location,
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

        logger.debug(f"[__getattribute__] Handling method {name}")
        if (
            issubclass(self.syft_action_data_type, ActionDataEmpty)
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
                syft_server_location=self.syft_server_location,
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
            logger.debug(f"Running method: {name}")

            def wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
                return _base_wrapper(*args, **kwargs)

            wrapper = types.MethodType(wrapper, type(self))
        else:
            logger.debug(f"Running non-method: {name}")

            wrapper = _base_wrapper

        try:
            wrapper.__doc__ = original_func.__doc__
            logger.debug(
                "Found original signature for ",
                name,
                inspect.signature(original_func),
            )
            wrapper.__ipython_inspector_signature_override__ = inspect.signature(
                original_func
            )
        except Exception:
            logger.debug(f"name={name} has no signature")

        # third party
        return wrapper

    def _syft_setattr(self, name: str, value: Any) -> Any:
        args = (name, value)
        kwargs: dict = {}
        op_name = "__setattr__"

        def fake_func(*args: Any, **kwargs: Any) -> Any:
            return ActionDataEmpty(syft_internal_type=self.syft_internal_type)

        if isinstance(
            self.syft_action_data_type, ActionDataEmpty
        ) or has_action_data_empty(args=args, kwargs=kwargs):
            local_func = fake_func
        else:
            local_func = getattr(self.syft_action_data, op_name)

        context = PreHookContext(
            obj=self,
            op_name=op_name,
            action_type=ActionType.SETATTRIBUTE,
            syft_server_location=self.syft_server_location,
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
        # bypass ipython canary verification
        if name == "_ipython_canary_method_should_not_exist_":
            return None

        # bypass certain attrs to prevent recursion issues
        if name.startswith("_syft") or name.startswith("syft"):
            return object.__getattribute__(self, name)

        if name in passthrough_attrs:
            return object.__getattribute__(self, name)

        # third party
        if name in self._syft_passthrough_attrs():
            return object.__getattribute__(self, name)

        # Handle bool operator on nonbools
        if name == "__bool__" and not self.syft_has_bool_attr:
            return self._syft_wrap_attribute_for_bool_on_nonbools(name)

        # check cache first
        if name in methods_to_check_in_cache:
            return getattr(self.syft_action_data_cache, name, None)

        # Handle Properties
        context_self = self._syft_get_attr_context(name)
        if self.syft_is_property(context_self, name):
            return self._syft_wrap_attribute_for_properties(name)

        # Handle anything else
        res = self._syft_wrap_attribute_for_methods(name)
        return res

    @property
    def is_link(self) -> bool:
        return self.syft_action_data_type is ActionDataLink

    def __setattr__(self, name: str, value: Any) -> Any:
        defined_on_self = name in self.__dict__ or name in self.__private_attributes__

        logger.debug(f">> {name} defined_on_self={defined_on_self}")

        # use the custom defined version
        if defined_on_self:
            self.__dict__[name] = value
            return value
        else:
            self._syft_setattr(name, value)
            context_self = self.syft_action_data
            return context_self.__setattr__(name, value)

    # def keys(self) -> KeysView[str]:
    #     return self.syft_action_data.keys()  # type: ignore

    ###### __DUNDER_MIFFLIN__

    # if we do not implement these boiler plate __method__'s then special infix
    # operations like x + y won't trigger __getattribute__
    # unless there is a super special reason we should write no code in these functions
    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        if self.is_mock:
            res = "TwinPointer(Mock)"
        elif self.is_real:
            res = "TwinPointer(Real)"
        elif not self.is_twin:
            res = "Pointer"

        if isinstance(self.syft_action_data_cache, ActionDataEmpty):
            data_repr_ = self.syft_action_data_repr_
        else:
            if inspect.isclass(self.syft_action_data_cache):
                data_repr_ = repr_cls(self.syft_action_data_cache)
            else:
                data_repr_ = (
                    self.syft_action_data_cache._repr_markdown_()
                    if (
                        self.syft_action_data_cache is not None
                        and hasattr(self.syft_action_data_cache, "_repr_markdown_")
                    )
                    else self.syft_action_data_cache.__repr__()
                )

        return f"\n**{res}**\n\n{data_repr_}\n"

    def _data_repr(self) -> str | None:
        if isinstance(self.syft_action_data_cache, ActionDataEmpty):
            data_repr = self.syft_action_data_repr_
        elif inspect.isclass(self.syft_action_data_cache):
            data_repr = repr_cls(self.syft_action_data_cache)
        else:
            data_repr = self.syft_action_data_cache.__repr__()

        return data_repr

    def __repr__(self) -> str:
        if self.is_mock:
            res = "TwinPointer(Mock)"
        elif self.is_real:
            res = "TwinPointer(Real)"
        if not self.is_twin:
            res = "Pointer"
        data_repr = self._data_repr()
        return f"{res}:\n{data_repr}"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__call__(*args, **kwds)

    def __str__(self) -> str:
        if not inspect.isclass(self):
            return self.__str__()
        else:
            return self.syft_action_data_str_

    def __len__(self) -> int:
        return self.__len__()

    def __hash__(self, *args: Any, **kwargs: Any) -> int:
        return super().__hash__(*args, **kwargs)

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

    def __ne__(self, other: Any) -> Any:
        return self._syft_output_action_object(self.__ne__(other))

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

    def __iter__(self) -> Any:
        return self._syft_output_action_object(self.__iter__())

    def __next__(self) -> Any:
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
    """
    This is a catch-all class for all objects that are not
    defined in the `action_types` dictionary.
    """

    __canonical_name__ = "AnyActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[type[Any]] = NoneType  # type: ignore
    # syft_passthrough_attrs: List[str] = []
    syft_dont_wrap_attrs: list[str] = ["__str__", "__repr__", "syft_action_data_str_"]
    syft_action_data_str_: str = ""

    def __float__(self) -> float:
        return float(self.syft_action_data)

    def __int__(self) -> float:
        return int(self.syft_action_data)


action_types[Any] = AnyActionObject


def debug_original_func(name: str, func: Callable) -> None:
    logger.debug(f"{name} func is:")
    logger.debug(f"inspect.isdatadescriptor = {inspect.isdatadescriptor(func)}")
    logger.debug(f"inspect.isgetsetdescriptor = {inspect.isgetsetdescriptor(func)}")
    logger.debug(f"inspect.isfunction = {inspect.isfunction(func)}")
    logger.debug(f"inspect.isbuiltin = {inspect.isbuiltin(func)}")
    logger.debug(f"inspect.ismethod = {inspect.ismethod(func)}")
    logger.debug(f"inspect.ismethoddescriptor = {inspect.ismethoddescriptor(func)}")


def is_action_data_empty(obj: Any) -> bool:
    return isinstance(obj, AnyActionObject) and issubclass(
        obj.syft_action_data_type, ActionDataEmpty
    )


def has_action_data_empty(args: Any, kwargs: Any) -> bool:
    for a in args:
        if is_action_data_empty(a):
            return True

    for a in kwargs.values():
        if is_action_data_empty(a):
            return True
    return False
