# future
from __future__ import annotations

# stdlib
from collections import OrderedDict
from collections.abc import Callable
import inspect
from inspect import Parameter
from inspect import signature
import types
from typing import Any
from typing import TYPE_CHECKING
from typing import cast
from typing import get_args
from typing import get_origin

# third party
from nacl.exceptions import BadSignatureError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import TypeAdapter

# relative
from ..abstract_server import AbstractServer
from ..protocol.data_protocol import PROTOCOL_TYPE
from ..protocol.data_protocol import get_data_protocol
from ..protocol.data_protocol import migrate_args_and_kwargs
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..serde.signature import Signature
from ..serde.signature import signature_remove
from ..server.credentials import SyftSigningKey
from ..server.credentials import SyftVerifyKey
from ..service.context import AuthedServiceContext
from ..service.context import ChangeContext
from ..service.metadata.server_metadata import ServerMetadataJSON
from ..service.response import SyftError
from ..service.response import SyftResponseMessage
from ..service.response import SyftSuccess
from ..service.service import UserLibConfigRegistry
from ..service.service import UserServiceConfigRegistry
from ..service.service import _format_signature
from ..service.service import _signature_error_message
from ..service.user.user_roles import ServiceRole
from ..service.warnings import APIEndpointWarning
from ..service.warnings import WarningContext
from ..types.errors import SyftException
from ..types.errors import exclude_from_traceback
from ..types.identity import Identity
from ..types.result import as_result
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftMigrationRegistry
from ..types.syft_object import SyftObject
from ..types.uid import LineageID
from ..types.uid import UID
from ..util.autoreload import autoreload_enabled
from ..util.markdown import as_markdown_python_code
from ..util.notebook_ui.components.tabulator_template import build_tabulator_table
from ..util.util import index_syft_by_module_name
from ..util.util import prompt_warning_message
from .connection import ServerConnection

if TYPE_CHECKING:
    # relative
    from ..server import Server
    from ..service.job.job_stash import Job


IPYNB_BACKGROUND_METHODS = {
    "getdoc",
    "_partialmethod",
    "__name__",
    "__code__",
    "__wrapped__",
    "__custom_documentations__",
    "__signature__",
    "__defaults__",
    "__kwdefaults__",
}

IPYNB_BACKGROUND_PREFIXES = ["_ipy", "_repr", "__ipython", "__pydantic"]


@exclude_from_traceback
def post_process_result(
    result: SyftError | SyftSuccess, unwrap_on_success: bool = False
) -> Any:
    if isinstance(result, SyftError):
        raise SyftException(public_message=result.message, server_trace=result.tb)

    if unwrap_on_success and isinstance(result, SyftSuccess):
        result = result.unwrap_value()

    return result


def _has_config_dict(t: Any) -> bool:
    return (
        # Use this instead of `issubclass`` to be compatible with python 3.10
        # `inspect.isclass(t) and issubclass(t, BaseModel)`` wouldn't work with
        # generics, e.g. `set[sy.UID]`, in python 3.10
        (hasattr(t, "__mro__") and BaseModel in t.__mro__)
        or hasattr(t, "__pydantic_config__")
    )


_config_dict = ConfigDict(arbitrary_types_allowed=True)


def _check_type(v: object, t: Any) -> Any:
    # TypeAdapter only accepts `config` arg if `t` does not
    # already contain a ConfigDict
    # i.e model_config in BaseModel and __pydantic_config__ in
    # other types.
    type_adapter = (
        TypeAdapter(t, config=_config_dict)
        if not _has_config_dict(t)
        else TypeAdapter(t)
    )

    return type_adapter.validate_python(v)


class APIRegistry:
    __api_registry__: dict[tuple, SyftAPI] = OrderedDict()

    @classmethod
    def set_api_for(
        cls,
        server_uid: UID | str,
        user_verify_key: SyftVerifyKey | str,
        api: SyftAPI,
    ) -> None:
        if isinstance(server_uid, str):
            server_uid = UID.from_string(server_uid)

        if isinstance(user_verify_key, str):
            user_verify_key = SyftVerifyKey.from_string(user_verify_key)

        key = (server_uid, user_verify_key)

        cls.__api_registry__[key] = api

    @classmethod
    @as_result(SyftException)
    def api_for(cls, server_uid: UID, user_verify_key: SyftVerifyKey) -> SyftAPI:
        key = (server_uid, user_verify_key)
        api_instance = cls.__api_registry__.get(key, None)

        if api_instance is None:
            msg = f"Unable to get the API. Please login to datasite {server_uid}"
            raise SyftException(public_message=msg)

        return api_instance

    @classmethod
    def get_all_api(cls) -> list[SyftAPI]:
        return list(cls.__api_registry__.values())

    @classmethod
    def get_by_recent_server_uid(cls, server_uid: UID) -> SyftAPI | None:
        for key, api in reversed(cls.__api_registry__.items()):
            if key[0] == server_uid:
                return api
        return None


@serializable()
class APIEndpoint(SyftObject):
    __canonical_name__ = "APIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    service_path: str
    module_path: str
    name: str
    description: str
    doc_string: str | None = None
    signature: Signature
    has_self: bool = False
    pre_kwargs: dict[str, Any] | None = None
    warning: APIEndpointWarning | None = None
    unwrap_on_success: bool = True


@serializable()
class LibEndpoint(SyftBaseObject):
    __canonical_name__ = "LibEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    # TODO: bad name, change
    service_path: str
    module_path: str
    name: str
    description: str
    doc_string: str | None = None
    signature: Signature
    has_self: bool = False
    pre_kwargs: dict[str, Any] | None = None


@serializable(attrs=["signature", "credentials", "serialized_message"])
class SignedSyftAPICall(SyftObject):
    __canonical_name__ = "SignedSyftAPICall"
    __version__ = SYFT_OBJECT_VERSION_1

    credentials: SyftVerifyKey
    signature: bytes
    serialized_message: bytes
    cached_deseralized_message: SyftAPICall | None = None

    @property
    def message(self) -> SyftAPICall:
        # from deserialize we might not have this attr because __init__ is skipped
        if not hasattr(self, "cached_deseralized_message"):
            self.cached_deseralized_message = None

        if self.cached_deseralized_message is None:
            self.cached_deseralized_message = _deserialize(
                blob=self.serialized_message, from_bytes=True
            )

        return self.cached_deseralized_message

    @property
    def is_valid(self) -> bool:
        try:
            _ = self.credentials.verify_key.verify(
                self.serialized_message, self.signature
            )
        except BadSignatureError:
            return False
        return True


@serializable()
class SyftAPICall(SyftObject):
    # version
    __canonical_name__ = "SyftAPICall"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    server_uid: UID
    path: str
    args: list
    kwargs: dict[str, Any]
    blocking: bool = True

    def sign(self, credentials: SyftSigningKey) -> SignedSyftAPICall:
        signed_message = credentials.signing_key.sign(_serialize(self, to_bytes=True))

        return SignedSyftAPICall(
            credentials=credentials.verify_key,
            serialized_message=signed_message.message,
            signature=signed_message.signature,
        )

    def __repr__(self) -> str:
        return f"SyftAPICall(path={self.path}, args={self.args}, kwargs={self.kwargs}, blocking={self.blocking})"


@serializable()
class SyftAPIData(SyftBaseObject):
    # version
    __canonical_name__ = "SyftAPIData"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    data: Any = None

    def sign(self, credentials: SyftSigningKey) -> SignedSyftAPICall:
        signed_message = credentials.signing_key.sign(_serialize(self, to_bytes=True))

        return SignedSyftAPICall(
            credentials=credentials.verify_key,
            serialized_message=signed_message.message,
            signature=signed_message.signature,
        )


class RemoteFunction(SyftObject):
    __canonical_name__ = "RemoteFunction"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = [
        "id",
        "server_uid",
        "signature",
        "path",
    ]

    server_uid: UID
    signature: Signature
    refresh_api_callback: Callable | None = None
    path: str
    make_call: Callable
    pre_kwargs: dict[str, Any] | None = None
    communication_protocol: PROTOCOL_TYPE
    warning: APIEndpointWarning | None = None
    custom_function: bool = False
    unwrap_on_success: bool = True

    @property
    def __ipython_inspector_signature_override__(self) -> Signature | None:
        return self.signature

    def prepare_args_and_kwargs(
        self, args: list | tuple, kwargs: dict[str, Any]
    ) -> tuple[tuple, dict[str, Any]]:
        # Validate and migrate args and kwargs
        res = validate_callable_args_and_kwargs(args, kwargs, self.signature).unwrap()
        args, kwargs = res

        args, kwargs = migrate_args_and_kwargs(
            to_protocol=self.communication_protocol, args=args, kwargs=kwargs
        )

        return tuple(args), kwargs

    def function_call(
        self, path: str, *args: Any, cache_result: bool = True, **kwargs: Any
    ) -> Any:
        if "blocking" in self.signature.parameters:
            raise Exception(
                f"Signature {self.signature} can't have 'blocking' kwarg because it's reserved"
            )

        blocking = True
        if "blocking" in kwargs:
            if path == "api.call_public_in_jobs":
                raise SyftException(
                    public_message="The 'blocking' parameter is not allowed for this function"
                )

            blocking = bool(kwargs["blocking"])
            del kwargs["blocking"]

        _valid_args, _valid_kwargs = self.prepare_args_and_kwargs(args, kwargs)
        if self.pre_kwargs:
            _valid_kwargs.update(self.pre_kwargs)

        _valid_kwargs["communication_protocol"] = self.communication_protocol

        api_call = SyftAPICall(
            server_uid=self.server_uid,
            path=path,
            args=list(_valid_args),
            kwargs=_valid_kwargs,
            blocking=blocking,
        )

        allowed = self.warning.show() if self.warning else True
        if not allowed:
            return
        result = self.make_call(api_call=api_call, cache_result=cache_result)

        # TODO: annotate this on the service method decorator
        API_CALLS_THAT_REQUIRE_REFRESH = ["settings.enable_eager_execution"]

        if path in API_CALLS_THAT_REQUIRE_REFRESH:
            if self.refresh_api_callback is not None:
                self.refresh_api_callback()

        result, _ = migrate_args_and_kwargs(
            [result], kwargs={}, to_latest_protocol=True
        )
        result = result[0]

        return post_process_result(result, self.unwrap_on_success)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function_call(self.path, *args, **kwargs)

    @property
    def mock(self) -> Any:
        if self.custom_function:
            remote_func = self

            class PrivateCustomAPIReference:
                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    return remote_func.function_call(
                        "api.call_public_in_jobs", *args, **kwargs
                    )

                @property
                def context(self) -> Any:
                    return remote_func.function_call("api.get_public_context")

            return PrivateCustomAPIReference()
        raise SyftException(
            public_message="This function doesn't support mock/private calls as it's not custom."
        )

    @property
    def private(self) -> Any:
        if self.custom_function:
            remote_func = self

            class PrivateCustomAPIReference:
                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    return remote_func.function_call(
                        "api.call_private_in_jobs", *args, **kwargs
                    )

                @property
                def context(self) -> Any:
                    return remote_func.function_call("api.get_private_context")

            return PrivateCustomAPIReference()
        raise SyftException(
            public_message="This function doesn't support mock/private calls as it's not custom."
        )

    @as_result(SyftException)
    def custom_function_actionobject_id(self) -> UID:
        if self.custom_function and self.pre_kwargs is not None:
            custom_path = self.pre_kwargs.get("path", "")
            api_call = SyftAPICall(
                server_uid=self.server_uid,
                path="api.view",
                args=[custom_path],
                kwargs={},
            )
            endpoint = self.make_call(api_call=api_call)
            if isinstance(endpoint, SyftSuccess):
                endpoint = endpoint.value
            return endpoint.action_object_id
        raise SyftException(public_message="This function is not a custom function")

    def _repr_markdown_(self, wrap_as_python: bool = False, indent: int = 0) -> str:
        if self.custom_function and self.pre_kwargs is not None:
            custom_path = self.pre_kwargs.get("path", "")
            api_call = SyftAPICall(
                server_uid=self.server_uid,
                path="api.view",
                args=[custom_path],
                kwargs={},
            )

            endpoint = self.make_call(api_call=api_call)
            if isinstance(endpoint, SyftSuccess):
                endpoint = endpoint.value

            str_repr = "## API: " + custom_path + "\n"
            if endpoint.description is not None:
                text = endpoint.description.text
            else:
                text = ""
            str_repr += (
                "### Description: "
                + f'<span style="font-weight: lighter;">{text}</span><br>'
                + "\n"
            )
            str_repr += "#### Private Code:\n"
            not_accessible_code = "N / A"
            private_code_repr = endpoint.private_function or not_accessible_code
            public_code_repr = endpoint.mock_function or not_accessible_code
            str_repr += as_markdown_python_code(private_code_repr) + "\n"
            if endpoint.private_helper_functions:
                str_repr += "##### Helper Functions:\n"
                for helper_function in endpoint.private_helper_functions:
                    str_repr += as_markdown_python_code(helper_function) + "\n"
            str_repr += "#### Public Code:\n"
            str_repr += as_markdown_python_code(public_code_repr) + "\n"
            if endpoint.mock_helper_functions:
                str_repr += "##### Helper Functions:\n"
                for helper_function in endpoint.mock_helper_functions:
                    str_repr += as_markdown_python_code(helper_function) + "\n"
            return str_repr
        return super()._repr_markdown_()


class RemoteUserCodeFunction(RemoteFunction):
    __canonical_name__ = "RemoteUserFunction"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = RemoteFunction.__repr_attrs__ + ["user_code_id"]

    api: SyftAPI

    def prepare_args_and_kwargs(
        self, args: list | tuple, kwargs: dict[str, Any]
    ) -> tuple[tuple, dict[str, Any]]:
        # relative
        from ..service.action.action_object import convert_to_pointers

        # Validate and migrate args and kwargs
        res = validate_callable_args_and_kwargs(args, kwargs, self.signature).unwrap()
        args, kwargs = res

        # Check remote function type to avoid function/method serialization
        # We can recover the function/method pointer by its UID in server side.
        for i in range(len(args)):
            if isinstance(args[i], RemoteFunction) and args[i].custom_function:
                args[i] = args[i].custom_function_id()  # type: ignore

        for k, v in kwargs.items():
            if isinstance(v, RemoteFunction) and v.custom_function:
                kwargs[k] = v.custom_function_actionobject_id().unwrap()

        args, kwargs = convert_to_pointers(
            api=self.api,
            server_uid=self.server_uid,
            args=args,
            kwargs=kwargs,
        )

        args, kwargs = migrate_args_and_kwargs(
            to_protocol=self.communication_protocol, args=args, kwargs=kwargs
        )

        return tuple(args), kwargs

    @property
    def user_code_id(self) -> UID | None:
        if self.pre_kwargs:
            return self.pre_kwargs.get("uid", None)
        else:
            return None

    @property
    def jobs(self) -> list[Job]:
        if self.user_code_id is None:
            raise SyftException(public_message="Could not find user_code_id")
        api_call = SyftAPICall(
            server_uid=self.server_uid,
            path="job.get_by_user_code_id",
            args=[self.user_code_id],
            kwargs={},
            blocking=True,
        )
        result = self.make_call(api_call=api_call)
        return post_process_result(result, self.unwrap_on_success)


def generate_remote_function(
    api: SyftAPI,
    server_uid: UID,
    signature: Signature,
    path: str,
    make_call: Callable,
    pre_kwargs: dict[str, Any] | None,
    communication_protocol: PROTOCOL_TYPE,
    warning: APIEndpointWarning | None,
    unwrap_on_success: bool = True,
) -> RemoteFunction:
    if "blocking" in signature.parameters:
        raise Exception(
            f"Signature {signature} can't have 'blocking' kwarg because it's reserved"
        )

    # UserCodes are always code.call with a user_code_id
    if path == "code.call" and pre_kwargs is not None and "uid" in pre_kwargs:
        remote_function = RemoteUserCodeFunction(
            api=api,
            server_uid=server_uid,
            signature=signature,
            path=path,
            make_call=make_call,
            pre_kwargs=pre_kwargs,
            communication_protocol=communication_protocol,
            warning=warning,
            user_code_id=pre_kwargs["uid"],
            unwrap_on_success=unwrap_on_success,
        )
    else:
        custom_function = bool(path == "api.call_in_jobs")
        remote_function = RemoteFunction(
            server_uid=server_uid,
            refresh_api_callback=api.refresh_api_callback,
            signature=signature,
            path=path,
            make_call=make_call,
            pre_kwargs=pre_kwargs,
            communication_protocol=communication_protocol,
            warning=warning,
            custom_function=custom_function,
            unwrap_on_success=unwrap_on_success,
        )

    return remote_function


def generate_remote_lib_function(
    api: SyftAPI,
    server_uid: UID,
    signature: Signature,
    path: str,
    module_path: str,
    make_call: Callable,
    communication_protocol: PROTOCOL_TYPE,
    pre_kwargs: dict[str, Any],
) -> Any:
    if "blocking" in signature.parameters:
        raise Exception(
            f"Signature {signature} can't have 'blocking' kwarg because its reserved"
        )

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # relative
        from ..service.action.action_object import TraceResultRegistry

        trace_result = TraceResultRegistry.get_trace_result_for_thread()

        if trace_result is not None:
            wrapper_make_call = trace_result.client.api.make_call  # type: ignore
            wrapper_server_uid = trace_result.client.api.server_uid  # type: ignore
        else:
            # somehow this is necessary to prevent shadowing problems
            wrapper_make_call = make_call
            wrapper_server_uid = server_uid
        blocking = True
        if "blocking" in kwargs:
            blocking = bool(kwargs["blocking"])
            del kwargs["blocking"]

        res = validate_callable_args_and_kwargs(args, kwargs, signature).unwrap()

        _valid_args, _valid_kwargs = res

        if pre_kwargs:
            _valid_kwargs.update(pre_kwargs)

        # relative
        from ..service.action.action_object import Action
        from ..service.action.action_object import ActionType
        from ..service.action.action_object import convert_to_pointers

        action_args, action_kwargs = convert_to_pointers(
            api, wrapper_server_uid, _valid_args, _valid_kwargs
        )

        # e.g. numpy.array -> numpy, array
        module, op = module_path.rsplit(".", 1)
        action = Action(
            path=module,
            op=op,
            remote_self=None,
            args=[x.syft_lineage_id for x in action_args],
            kwargs={k: v.syft_lineage_id for k, v in action_kwargs},
            action_type=ActionType.FUNCTION,
            # TODO: fix
            result_id=LineageID(UID(), 1),
        )
        service_args = [action]
        # TODO: implement properly
        if trace_result is not None:
            trace_result.result += [action]

        api_call = SyftAPICall(
            server_uid=wrapper_server_uid,
            path=path,
            args=service_args,
            kwargs={},
            blocking=blocking,
        )

        result = wrapper_make_call(api_call=api_call)
        result = post_process_result(result, unwrap_on_success=True)

        return result

    wrapper.__ipython_inspector_signature_override__ = signature
    return wrapper


class APISubModulesView(SyftObject):
    __canonical_name__ = "APISubModulesView"
    __version__ = SYFT_OBJECT_VERSION_1

    submodule: str = ""
    endpoints: list[str] = []

    __syft_include_id_coll_repr__ = False

    def _coll_repr_(self) -> dict[str, Any]:
        return {"submodule": self.submodule, "endpoints": "\n".join(self.endpoints)}


@serializable(canonical_name="APIModule", version=1)
class APIModule:
    _modules: list[str]
    path: str
    refresh_callback: Callable | None

    def __init__(self, path: str, refresh_callback: Callable | None) -> None:
        self._modules = []
        self.path = path
        self.refresh_callback = refresh_callback

    def __dir__(self) -> list[str]:
        return self._modules + ["path"]

    def has_submodule(self, name: str) -> bool:
        """We use this as hasattr() triggers __getattribute__ which triggers recursion"""
        try:
            _ = object.__getattribute__(self, name)
            return True
        except AttributeError:
            return False

    def _add_submodule(
        self, attr_name: str, module_or_func: Callable | APIModule
    ) -> None:
        setattr(self, attr_name, module_or_func)
        self._modules.append(attr_name)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # if we fail, we refresh the api and try again
            # however, we dont want this to happen all the time because of ipy magic happening
            # in the background
            if (
                self.refresh_callback is not None
                and name not in IPYNB_BACKGROUND_METHODS
                and not any(
                    name.startswith(prefix) for prefix in IPYNB_BACKGROUND_PREFIXES
                )
            ):
                api = self.refresh_callback()
                try:
                    # get current path in the module tree
                    new_current_module = api.services
                    for submodule in self.path.split("."):
                        if submodule != "":
                            new_current_module = getattr(new_current_module, submodule)
                    # retry getting the attribute, if this fails, we throw an error
                    return object.__getattribute__(new_current_module, name)
                except AttributeError:
                    pass
            raise AttributeError(
                f"'APIModule' api{self.path} object has no submodule or method '{name}', "
                "you may not have permission to access the module you are trying to access."
                "If you think this is an error, try calling `client.refresh()` to update the API."
            )

    def __getitem__(self, key: str | int) -> Any:
        if hasattr(self, "get_index"):
            return self.get_index(key)
        if hasattr(self, "get_all"):
            return self.get_all()[key]
        raise NotImplementedError

    def __iter__(self) -> Any:
        if hasattr(self, "get_all"):
            return iter(self.get_all())
        raise NotImplementedError

    def _repr_html_(self) -> Any:
        if self.path == "settings":
            return self.get()._repr_html_()

        if not hasattr(self, "get_all"):

            def recursively_get_submodules(
                module: APIModule | Callable,
            ) -> list[APIModule | Callable]:
                children = [module]
                if isinstance(module, APIModule):
                    for submodule_name in module._modules:
                        submodule = getattr(module, submodule_name)
                        children += recursively_get_submodules(submodule)
                return children

            views = []
            for submodule_name in self._modules:
                submodule = getattr(self, submodule_name)
                children = recursively_get_submodules(submodule)
                child_paths = [
                    x.path for x in children if isinstance(x, RemoteFunction)
                ]
                views.append(
                    APISubModulesView(submodule=submodule_name, endpoints=child_paths)
                )

            return build_tabulator_table(views)

        # should never happen?
        results = self.get_all()
        return results._repr_html_()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return NotImplementedError


# TODO ERROR: what is this function return type???
@as_result(SyftException)
def debox_signed_syftapicall_response(
    signed_result: SignedSyftAPICall | Any,
) -> Any:
    if not isinstance(signed_result, SignedSyftAPICall):
        raise SyftException(public_message="The result is not signed")

    if not signed_result.is_valid:
        raise SyftException(public_message="The result signature is invalid")

    return signed_result.message.data


def downgrade_signature(signature: Signature, object_versions: dict) -> Signature:
    migrated_parameters = []
    for parameter in signature.parameters.values():
        annotation = unwrap_and_migrate_annotation(
            parameter.annotation, object_versions
        )
        migrated_parameter = Parameter(
            name=parameter.name,
            default=parameter.default,
            annotation=annotation,
            kind=parameter.kind,
        )
        migrated_parameters.append(migrated_parameter)

    migrated_return_annotation = unwrap_and_migrate_annotation(
        signature.return_annotation, object_versions
    )

    try:
        new_signature = Signature(
            parameters=migrated_parameters,
            return_annotation=migrated_return_annotation,
        )
    except Exception as e:
        raise e

    return new_signature


def unwrap_and_migrate_annotation(annotation: Any, object_versions: dict) -> Any:
    args = get_args(annotation)
    origin = get_origin(annotation)
    if len(args) == 0:
        if (
            isinstance(annotation, type)
            and issubclass(annotation, SyftBaseObject)
            and annotation.__canonical_name__ in object_versions
        ):
            downgrade_to_version = int(
                max(object_versions[annotation.__canonical_name__])
            )
            downgrade_klass_name = SyftMigrationRegistry.__migration_version_registry__[
                annotation.__canonical_name__
            ][downgrade_to_version]
            new_arg = index_syft_by_module_name(downgrade_klass_name)
            return new_arg
        else:
            return annotation

    migrated_annotations = []
    for arg in args:
        migrated_annotation = unwrap_and_migrate_annotation(arg, object_versions)
        migrated_annotations.append(migrated_annotation)

    migrated_annotations_tuple = tuple(migrated_annotations)

    if hasattr(annotation, "copy_with"):
        return annotation.copy_with(migrated_annotations_tuple)
    elif origin is not None:
        return origin[migrated_annotations_tuple]
    else:
        return migrated_annotation[0]


def result_needs_api_update(api_call_result: Any) -> bool:
    # relative
    from ..service.request.request import Request
    from ..service.request.request import UserCodeStatusChange

    if isinstance(api_call_result, Request) and any(
        isinstance(x, UserCodeStatusChange) for x in api_call_result.changes
    ):
        return True
    if isinstance(api_call_result, SyftSuccess) and api_call_result.require_api_update:
        return True
    return False


@serializable(
    attrs=[
        "endpoints",
        "server_uid",
        "server_name",
        "lib_endpoints",
        "communication_protocol",
    ]
)
class SyftAPI(SyftObject):
    # version
    __canonical_name__ = "SyftAPI"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    connection: ServerConnection | None = None
    server_uid: UID | None = None
    server_name: str | None = None
    endpoints: dict[str, APIEndpoint]
    lib_endpoints: dict[str, LibEndpoint] | None = None
    api_module: APIModule | None = None
    libs: APIModule | None = None
    signing_key: SyftSigningKey | None = None
    # serde / storage rules
    refresh_api_callback: Callable | None = None
    __user_role: ServiceRole = ServiceRole.NONE
    communication_protocol: PROTOCOL_TYPE
    metadata: ServerMetadataJSON | None = None

    # informs getattr does not have nasty side effects
    __syft_allow_autocomplete__ = ["services"]

    def __dir__(self) -> list[str]:
        modules = getattr(self.api_module, "_modules", [])
        return ["services"] + modules

    def __syft_dir__(self) -> list[str]:
        modules = getattr(self.api_module, "_modules", [])
        return ["services"] + modules

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.api_module, name)
        except Exception:
            raise AttributeError(
                f"'SyftAPI' object has no submodule or method '{name}', "
                "you may not have permission to access the module you are trying to access."
                "If you think this is an error, try calling `client.refresh()` to update the API."
            )

    @staticmethod
    def for_user(
        server: AbstractServer,
        communication_protocol: PROTOCOL_TYPE,
        user_verify_key: SyftVerifyKey | None = None,
    ) -> SyftAPI:
        # relative
        from ..service.api.api_service import APIService

        # TODO: Maybe there is a possibility of merging ServiceConfig and APIEndpoint
        from ..service.code.user_code_service import UserCodeService

        # find user role by verify_key
        # TODO: we should probably not allow empty verify keys but instead make user always register
        role = server.get_role_for_credentials(user_verify_key)
        _user_service_config_registry = UserServiceConfigRegistry.from_role(role)
        _user_lib_config_registry = UserLibConfigRegistry.from_user(user_verify_key)
        endpoints: dict[str, APIEndpoint] = {}
        lib_endpoints: dict[str, LibEndpoint] = {}
        warning_context = WarningContext(
            server=server, role=role, credentials=user_verify_key
        )

        # If server uses a higher protocol version than client, then
        # signatures needs to be downgraded.
        if server.current_protocol == "dev" and communication_protocol != "dev":
            # We assume dev is the highest staged protocol
            signature_needs_downgrade = True
        else:
            signature_needs_downgrade = server.current_protocol != "dev" and int(
                server.current_protocol
            ) > int(communication_protocol)
        data_protocol = get_data_protocol()

        if signature_needs_downgrade:
            object_version_for_protocol = data_protocol.get_object_versions(
                communication_protocol
            )

        for (
            path,
            service_config,
        ) in _user_service_config_registry.get_registered_configs().items():
            if not service_config.is_from_lib:
                service_warning = service_config.warning
                if service_warning:
                    service_warning = service_warning.message_from(warning_context)
                    service_warning.enabled = server.enable_warnings

                signature = (
                    downgrade_signature(
                        signature=service_config.signature,
                        object_versions=object_version_for_protocol,
                    )
                    if signature_needs_downgrade
                    else service_config.signature
                )

                endpoint = APIEndpoint(
                    service_path=path,
                    module_path=path,
                    name=service_config.public_name,
                    description="",
                    doc_string=service_config.doc_string,
                    signature=signature,  # TODO: Migrate signature based on communication protocol
                    has_self=False,
                    warning=service_warning,
                    unwrap_on_success=service_config.unwrap_on_success,
                )
                endpoints[path] = endpoint

        for (
            path,
            lib_config,
        ) in _user_lib_config_registry.get_registered_configs().items():
            endpoint = LibEndpoint(
                service_path="action.execute",
                module_path=path,
                name=lib_config.public_name,
                description="",
                doc_string=lib_config.doc_string,
                signature=lib_config.signature,
                has_self=False,
            )
            lib_endpoints[path] = endpoint

        # 🟡 TODO 35: fix root context
        context = AuthedServiceContext(server=server, credentials=user_verify_key)
        method = server.get_method_with_context(
            UserCodeService.get_all_for_user, context
        )
        code_items = method()

        for code_item in code_items:
            path = "code.call"
            unique_path = f"code.call_{code_item.service_func_name}"
            endpoint = APIEndpoint(
                service_path=path,
                module_path=path,
                name=code_item.service_func_name,
                description="",
                doc_string=f"Users custom func {code_item.service_func_name}",
                signature=code_item.signature,
                has_self=False,
                pre_kwargs={"uid": code_item.id},
            )
            endpoints[unique_path] = endpoint

        # get admin defined custom api endpoints
        method = server.get_method_with_context(APIService.get_endpoints, context)
        custom_endpoints = method().unwrap()
        for custom_endpoint in custom_endpoints:
            pre_kwargs = {"path": custom_endpoint.path}
            service_path = "api.call_in_jobs"
            path = custom_endpoint.path
            api_end = custom_endpoint.path.split(".")[-1]
            endpoint = APIEndpoint(
                service_path=service_path,
                module_path=path,
                name=api_end,
                description="",
                doc_string="",
                signature=custom_endpoint.signature,
                has_self=False,
                pre_kwargs=pre_kwargs,
            )
            endpoints[path] = endpoint

        return SyftAPI(
            server_name=server.name,
            server_uid=server.id,
            endpoints=endpoints,
            lib_endpoints=lib_endpoints,
            __user_role=role,
            communication_protocol=communication_protocol,
        )

    @property
    def user_role(self) -> ServiceRole:
        return self.__user_role

    def make_call(self, api_call: SyftAPICall, cache_result: bool = True) -> Any:
        signed_call = api_call.sign(credentials=self.signing_key)
        if self.connection is not None:
            signed_result = self.connection.make_call(signed_call)
        else:
            raise SyftException(public_message="API connection is None")

        result = debox_signed_syftapicall_response(signed_result=signed_result).unwrap()
        if isinstance(result, SyftResponseMessage):
            for warning in result.client_warnings:
                prompt_warning_message(
                    message=warning,
                )
        # we update the api when we create objects that change it
        self.update_api(result)
        return result

    def update_api(self, api_call_result: Any) -> None:
        # TODO: hacky stuff with typing and imports to prevent circular imports
        if result_needs_api_update(api_call_result):
            if self.refresh_api_callback is not None:
                self.refresh_api_callback()

    def _add_route(
        self, api_module: APIModule, endpoint: APIEndpoint, endpoint_method: Callable
    ) -> None:
        """Recursively create a module path to the route endpoint."""

        _modules = endpoint.module_path.split(".")[:-1] + [endpoint.name]

        _self = api_module
        _last_module = _modules.pop()
        while _modules:
            module = _modules.pop(0)
            if not _self.has_submodule(module):
                submodule_path = (
                    f"{_self.path}.{module}" if _self.path != "" else module
                )
                _self._add_submodule(
                    module,
                    APIModule(
                        path=submodule_path, refresh_callback=self.refresh_api_callback
                    ),
                )
            _self = getattr(_self, module)
        _self._add_submodule(_last_module, endpoint_method)

    def generate_endpoints(self) -> None:
        def build_endpoint_tree(
            endpoints: dict[str, LibEndpoint | APIEndpoint],
            communication_protocol: PROTOCOL_TYPE,
        ) -> APIModule:
            api_module = APIModule(path="", refresh_callback=self.refresh_api_callback)
            for v in endpoints.values():
                signature = v.signature
                args_to_remove = ["context"]
                if not v.has_self:
                    args_to_remove.append("self")
                signature = signature_remove(signature, args_to_remove)
                if isinstance(v, APIEndpoint):
                    endpoint_function = generate_remote_function(
                        self,
                        self.server_uid,
                        signature,
                        v.service_path,
                        self.make_call,
                        pre_kwargs=v.pre_kwargs,
                        warning=v.warning,
                        communication_protocol=communication_protocol,
                        unwrap_on_success=v.unwrap_on_success,
                    )
                elif isinstance(v, LibEndpoint):
                    endpoint_function = generate_remote_lib_function(
                        self,
                        self.server_uid,
                        signature,
                        v.service_path,
                        v.module_path,
                        self.make_call,
                        pre_kwargs=v.pre_kwargs,
                        communication_protocol=communication_protocol,
                    )

                endpoint_function.__doc__ = v.doc_string
                self._add_route(api_module, v, endpoint_function)
            return api_module

        if self.lib_endpoints is not None:
            self.libs = build_endpoint_tree(
                self.lib_endpoints, self.communication_protocol
            )
        self.api_module = build_endpoint_tree(
            self.endpoints, self.communication_protocol
        )

    @property
    def services(self) -> APIModule:
        if self.api_module is None:
            self.generate_endpoints()
        return cast(APIModule, self.api_module)

    @property
    def lib(self) -> APIModule:
        if self.libs is None:
            self.generate_endpoints()
        return cast(APIModule, self.libs)

    def has_service(self, service_name: str) -> bool:
        return hasattr(self.services, service_name)

    def has_lib(self, lib_name: str) -> bool:
        return hasattr(self.lib, lib_name)

    def __repr__(self) -> str:
        modules = self.services
        _repr_str = "client.api.services\n"
        if modules is not None:
            for attr_name in modules._modules:
                module_or_func = getattr(modules, attr_name)
                module_path_str = f"client.api.services.{attr_name}"
                _repr_str += f"\n{module_path_str}\n\n"
                if hasattr(module_or_func, "_modules"):
                    for func_name in module_or_func._modules:
                        func = getattr(module_or_func, func_name)
                        sig = getattr(
                            func, "__ipython_inspector_signature_override__", ""
                        )
                        _repr_str += f"{module_path_str}.{func_name}{sig}\n\n"
        return _repr_str


# code from here:
# https://github.com/ipython/ipython/blob/339c0d510a1f3cb2158dd8c6e7f4ac89aa4c89d8/IPython/core/oinspect.py#L370
def _render_signature(obj_signature: Signature, obj_name: str) -> str:
    """
    This was mostly taken from inspect.Signature.__str__.
    Look there for the comments.
    The only change is to add linebreaks when this gets too long.
    """
    result = []
    pos_only = False
    kw_only = True
    for param in obj_signature.parameters.values():
        if param.kind == inspect._POSITIONAL_ONLY:
            pos_only = True
        elif pos_only:
            result.append("/")
            pos_only = False

        if param.kind == inspect._VAR_POSITIONAL:
            kw_only = False
        elif param.kind == inspect._KEYWORD_ONLY and kw_only:
            result.append("*")
            kw_only = False

        result.append(str(param))

    if pos_only:
        result.append("/")

    # add up name, parameters, braces (2), and commas
    if len(obj_name) + sum(len(r) + 2 for r in result) > 75:
        # This doesn’t fit behind “Signature: ” in an inspect window.
        rendered = "{}(\n{})".format(obj_name, "".join(f"    {r},\n" for r in result))
    else:
        rendered = "{}({})".format(obj_name, ", ".join(result))

    if obj_signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(obj_signature.return_annotation)
        rendered += f" -> {anno}"

    return rendered


def _getdef(self: Any, obj: Any, oname: str = "") -> str | None:
    """Return the call signature for any callable object.
    If any exception is generated, None is returned instead and the
    exception is suppressed."""
    try:
        return _render_signature(signature(obj), oname)
    except:  # noqa: E722
        return None


def monkey_patch_getdef(self: Any, obj: Any, oname: str = "") -> str | None:
    try:
        if hasattr(obj, "__ipython_inspector_signature_override__"):
            return _render_signature(
                obj.__ipython_inspector_signature_override__, oname
            )
        return _getdef(self, obj, oname)
    except Exception:
        return None


# try to monkeypatch IPython
try:
    # third party
    from IPython.core.oinspect import Inspector

    if not hasattr(Inspector, "_getdef_bak"):
        Inspector._getdef_bak = Inspector._getdef
        Inspector._getdef = types.MethodType(monkey_patch_getdef, Inspector)
except Exception:
    pass  # nosec


@serializable(canonical_name="ServerIdentity", version=1)
class ServerIdentity(Identity):
    server_name: str

    @staticmethod
    def from_api(api: SyftAPI) -> ServerIdentity:
        # stores the name root verify key of the datasite server
        if api.connection is None:
            raise ValueError(
                "{api}'s connection is None. Can't get the server identity"
            )
        server_metadata = api.connection.get_server_metadata(api.signing_key)
        return ServerIdentity(
            server_name=server_metadata.name,
            server_id=api.server_uid,
            verify_key=SyftVerifyKey.from_string(server_metadata.verify_key),
        )

    @classmethod
    def from_change_context(cls, context: ChangeContext) -> ServerIdentity:
        if context.server is None:
            raise ValueError(f"{context}'s server is None")
        return cls(
            server_name=context.server.name,
            server_id=context.server.id,
            verify_key=context.server.signing_key.verify_key,
        )

    @classmethod
    def from_server(cls, server: Server) -> ServerIdentity:
        return cls(
            server_name=server.name,
            server_id=server.id,
            verify_key=server.signing_key.verify_key,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ServerIdentity):
            return False
        return (
            self.server_name == other.server_name
            and self.verify_key == other.verify_key
            and self.server_id == other.server_id
        )

    def __hash__(self) -> int:
        return hash((self.server_name, self.verify_key))

    def __repr__(self) -> str:
        return f"ServerIdentity <name={self.server_name}, id={self.server_id.short()}, 🔑={str(self.verify_key)[0:8]}>"


@as_result(SyftException)
def validate_callable_args_and_kwargs(
    args: list, kwargs: dict, signature: Signature
) -> tuple[list, dict]:
    _valid_kwargs = {}
    if "kwargs" in signature.parameters:
        _valid_kwargs = kwargs
    else:
        for key, value in kwargs.items():
            if key not in signature.parameters:
                valid_parameters = list(signature.parameters)
                valid_parameters_msg = (
                    f"Valid parameter: {valid_parameters}"
                    if len(valid_parameters) == 1
                    else f"Valid parameters: {valid_parameters}"
                )

                raise SyftException(
                    public_message=(
                        f"Invalid parameter: `{key}`\n"
                        f"{valid_parameters_msg}\n"
                        f"{_signature_error_message(_format_signature(signature))}"
                    )
                )
            param = signature.parameters[key]
            if isinstance(param.annotation, str):
                # 🟡 TODO 21: make this work for weird string type situations
                # happens when from __future__ import annotations in a class file
                t = index_syft_by_module_name(param.annotation)
            else:
                t = param.annotation

            if t is not inspect.Parameter.empty:
                try:
                    _check_type(value, t)
                except ValueError:
                    # TODO: fix this properly
                    if not (t == type(Any)):
                        _type_str = getattr(t, "__name__", str(t))
                        raise SyftException(
                            public_message=f"`{key}` must be of type `{_type_str}` not `{type(value).__name__}`"
                            f"{_signature_error_message(_format_signature(signature))}"
                        )

            _valid_kwargs[key] = value

    # signature.parameters is an OrderedDict, therefore,
    # its fair to assume that order of args
    # and the signature.parameters should always match
    _valid_args = []
    if "args" in signature.parameters:
        _valid_args = args
    else:
        for (param_key, param), arg in zip(signature.parameters.items(), args):
            if param_key in _valid_kwargs:
                continue
            t = param.annotation
            msg = None
            try:
                if t is not inspect.Parameter.empty:
                    _check_type(arg, t)
            except ValueError:
                t_arg = type(arg)
                if (
                    autoreload_enabled()
                    and t.__module__ == t_arg.__module__
                    and t.__name__ == t_arg.__name__
                ):
                    # ignore error when autoreload_enabled()
                    pass
                else:
                    _type_str = getattr(t, "__name__", str(t))

                    msg = (
                        f"Arg is `{arg}`. \nIt must be of type `{_type_str}`, not `{type(arg).__name__}`\n"
                        f"{_signature_error_message(_format_signature(signature))}"
                    )

            if msg:
                raise SyftException(public_message=msg)

            _valid_args.append(arg)

    return _valid_args, _valid_kwargs


RemoteFunction.model_rebuild(force=True)
RemoteUserCodeFunction.model_rebuild(force=True)
