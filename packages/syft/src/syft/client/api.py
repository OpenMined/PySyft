# future
from __future__ import annotations

# stdlib
import inspect
from inspect import signature
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import _GenericAlias

# third party
from nacl.exceptions import BadSignatureError
from pydantic import BaseModel
from pydantic import EmailStr
from result import OkErr
from result import Result
from typeguard import check_type

# relative
from ..abstract_node import AbstractNode
from ..node.credentials import SyftSigningKey
from ..node.credentials import SyftVerifyKey
from ..serde.deserialize import _deserialize
from ..serde.recursive import index_syft_by_module_name
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..serde.signature import Signature
from ..serde.signature import signature_remove_context
from ..serde.signature import signature_remove_self
from ..service.context import AuthedServiceContext
from ..service.response import SyftAttributeError
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.service import UserServiceConfigRegistry
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.uid import UID
from ..util.telemetry import instrument
from .connection import NodeConnection


class APIRegistry:
    __api_registry__: Dict[str, SyftAPI] = {}

    @classmethod
    def set_api_for(cls, node_uid: Union[UID, str], api: SyftAPI) -> None:
        if isinstance(node_uid, str):
            node_uid = UID.from_string(node_uid)
        cls.__api_registry__[node_uid] = api

    @classmethod
    def api_for(cls, node_uid: UID) -> SyftAPI:
        return cls.__api_registry__[node_uid]

    @classmethod
    def get_all_api(cls) -> List[SyftAPI]:
        return list(cls.__api_registry__.values())


@serializable()
class APIEndpoint(SyftBaseObject):
    path: str
    name: str
    description: str
    doc_string: Optional[str]
    signature: Signature
    has_self: bool = False
    pre_kwargs: Optional[Dict[str, Any]]


@serializable(attrs=["signature", "credentials", "serialized_message"])
class SignedSyftAPICall(SyftObject):
    __canonical_name__ = "SignedSyftAPICall"
    __version__ = SYFT_OBJECT_VERSION_1

    credentials: SyftVerifyKey
    signature: bytes
    serialized_message: bytes
    cached_deseralized_message: Optional[SyftAPICall] = None

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
    def is_valid(self) -> Result[SyftSuccess, SyftError]:
        try:
            _ = self.credentials.verify_key.verify(
                self.serialized_message, self.signature
            )
        except BadSignatureError:
            return SyftError(message="BadSignatureError")

        return SyftSuccess(message="Credentials are valid")


@instrument
@serializable()
class SyftAPICall(SyftObject):
    # version
    __canonical_name__ = "SyftAPICall"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    node_uid: UID
    path: str
    args: List
    kwargs: Dict[str, Any]
    blocking: bool = True

    def sign(self, credentials: SyftSigningKey) -> SignedSyftAPICall:
        signed_message = credentials.signing_key.sign(_serialize(self, to_bytes=True))

        return SignedSyftAPICall(
            credentials=credentials.verify_key,
            serialized_message=signed_message.message,
            signature=signed_message.signature,
        )


@instrument
@serializable()
class SyftAPIData(SyftBaseObject):
    # version
    __canonical_name__ = "SyftAPIData"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    data: Any

    def sign(self, credentials: SyftSigningKey) -> SignedSyftAPICall:
        signed_message = credentials.signing_key.sign(_serialize(self, to_bytes=True))

        return SignedSyftAPICall(
            credentials=credentials.verify_key,
            serialized_message=signed_message.message,
            signature=signed_message.signature,
        )


def generate_remote_function(
    node_uid: UID,
    signature: Signature,
    path: str,
    make_call: Callable,
    pre_kwargs: Dict[str, Any],
):
    if "blocking" in signature.parameters:
        raise Exception(
            f"Signature {signature} can't have 'blocking' kwarg because its reserved"
        )

    def wrapper(*args, **kwargs):
        blocking = True
        if "blocking" in kwargs:
            blocking = bool(kwargs["blocking"])
            del kwargs["blocking"]

        _valid_kwargs = {}
        if "kwargs" in signature.parameters:
            _valid_kwargs = kwargs
        else:
            for key, value in kwargs.items():
                if key not in signature.parameters:
                    return SyftError(
                        message=f"""Invalid parameter: `{key}`. Valid Parameters: {list(signature.parameters)}"""
                    )
                param = signature.parameters[key]
                if isinstance(param.annotation, str):
                    # 🟡 TODO 21: make this work for weird string type situations
                    # happens when from __future__ import annotations in a class file
                    t = index_syft_by_module_name(param.annotation)
                else:
                    t = param.annotation
                msg = None
                try:
                    if t is not inspect.Parameter.empty:
                        if isinstance(t, _GenericAlias) and type(None) in t.__args__:
                            for v in t.__args__:
                                if issubclass(v, EmailStr):
                                    v = str
                                check_type(key, value, v)  # raises Exception
                                break  # only need one to match
                        else:
                            check_type(key, value, t)  # raises Exception
                except TypeError:
                    _type_str = getattr(t, "__name__", str(t))
                    msg = f"`{key}` must be of type `{_type_str}` not `{type(value).__name__}`"

                if msg:
                    return SyftError(message=msg)

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
                        if isinstance(t, _GenericAlias) and type(None) in t.__args__:
                            for v in t.__args__:
                                if issubclass(v, EmailStr):
                                    v = str
                                check_type(param_key, arg, v)  # raises Exception
                                break  # only need one to match
                        else:
                            check_type(param_key, arg, t)  # raises Exception
                except TypeError:
                    _type_str = getattr(t, "__name__", str(t))
                    msg = f"Arg: {arg} must be {_type_str} not {type(arg).__name__}"
                if msg:
                    return SyftError(message=msg)

                _valid_args.append(arg)

        if pre_kwargs:
            _valid_kwargs.update(pre_kwargs)
        api_call = SyftAPICall(
            node_uid=node_uid,
            path=path,
            args=_valid_args,
            kwargs=_valid_kwargs,
            blocking=blocking,
        )
        result = make_call(api_call=api_call)
        return result

    wrapper.__ipython_inspector_signature_override__ = signature
    return wrapper


@serializable()
class APIModule:
    _modules: List[APIModule]
    path: str

    def __init__(self, path: str) -> None:
        self._modules = []
        self.path = path

    def _add_submodule(
        self, attr_name: str, module_or_func: Union[Callable, APIModule]
    ):
        setattr(self, attr_name, module_or_func)
        self._modules.append(attr_name)

    def __getattribute__(self, name: str):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            raise SyftAttributeError(
                f"'APIModule' api{self.path} object has no submodule or method '{name}', "
                "you may not have permission to access the module you are trying to access"
            )

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int) and hasattr(self, "get_all"):
            return self.get_all()[key]
        raise NotImplementedError

    def _repr_html_(self) -> Any:
        if not hasattr(self, "get_all"):
            return NotImplementedError
        results = self.get_all()
        return results._repr_html_()


@instrument
@serializable(attrs=["endpoints", "node_uid", "node_name"])
class SyftAPI(SyftObject):
    # version
    __canonical_name__ = "SyftAPI"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    connection: Optional[NodeConnection] = None
    node_uid: Optional[UID] = None
    node_name: Optional[str] = None
    endpoints: Dict[str, APIEndpoint]
    api_module: Optional[APIModule] = None
    signing_key: Optional[SyftSigningKey] = None
    # serde / storage rules
    refresh_api_callback: Optional[Callable] = None

    # def __post_init__(self) -> None:
    #     pass

    @staticmethod
    def for_user(
        node: AbstractNode, user_verify_key: Optional[SyftVerifyKey] = None
    ) -> SyftAPI:
        # relative
        # TODO: Maybe there is a possibility of merging ServiceConfig and APIEndpoint
        from ..service.code.user_code_service import UserCodeService

        # find user role by verify_key
        # TODO: we should probably not allow empty verify keys but instead make user always register
        role = node.get_role_for_credentials(user_verify_key)
        _user_service_config_registry = UserServiceConfigRegistry.from_role(role)
        endpoints = {}

        for (
            path,
            service_config,
        ) in _user_service_config_registry.get_registered_configs().items():
            endpoint = APIEndpoint(
                path=path,
                name=service_config.public_name,
                description="",
                doc_string=service_config.doc_string,
                signature=service_config.signature,
                has_self=False,
            )
            endpoints[path] = endpoint

        # 🟡 TODO 35: fix root context
        context = AuthedServiceContext(credentials=user_verify_key)
        method = node.get_method_with_context(UserCodeService.get_all_for_user, context)
        code_items = method()

        for code_item in code_items:
            api_path = "code.call"
            unique_path = f"code.call_{code_item.service_func_name}"
            endpoint = APIEndpoint(
                path=api_path,
                name=code_item.service_func_name,
                description="",
                doc_string=f"Users custom func {code_item.service_func_name}",
                signature=code_item.signature,
                has_self=False,
                pre_kwargs={"uid": code_item.id},
            )
            endpoints[unique_path] = endpoint

        return SyftAPI(node_name=node.name, node_uid=node.id, endpoints=endpoints)

    def make_call(self, api_call: SyftAPICall) -> Result:
        signed_call = api_call.sign(credentials=self.signing_key)
        signed_result = self.connection.make_call(signed_call)

        if not isinstance(signed_result, SignedSyftAPICall):
            return SyftError(message="The result is not signed")  # type: ignore

        if not signed_result.is_valid:
            return SyftError(message="The result signature is invalid")  # type: ignore

        result = signed_result.message.data

        if isinstance(result, OkErr):
            if result.is_ok():
                res = result.ok()
                # we update the api when we create objects that change it
                self.update_api(res)
                return res
            else:
                return result.err()
        return result

    def update_api(self, api_call_result):
        # TODO: hacky stuff with typing and imports to prevent circular imports
        # relative
        from ..service.request.request import Request
        from ..service.request.request import UserCodeStatusChange

        if isinstance(api_call_result, Request) and any(
            [isinstance(x, UserCodeStatusChange) for x in api_call_result.changes]
        ):
            if self.refresh_api_callback is not None:
                self.refresh_api_callback()

    @staticmethod
    def _add_route(
        api_module: APIModule, endpoint: APIEndpoint, endpoint_method: Callable
    ):
        """Recursively create a module path to the route endpoint."""

        _modules = endpoint.path.split(".")[:-1] + [endpoint.name]

        _self = api_module
        _last_module = _modules.pop()
        while _modules:
            module = _modules.pop(0)
            if not hasattr(_self, module):
                submodule_path = f"{_self.path}.{module}"
                _self._add_submodule(module, APIModule(path=submodule_path))
            _self = getattr(_self, module)
        _self._add_submodule(_last_module, endpoint_method)

    def generate_endpoints(self) -> None:
        api_module = APIModule(path="")
        for _, v in self.endpoints.items():
            signature = v.signature
            if not v.has_self:
                signature = signature_remove_self(signature)
            signature = signature_remove_context(signature)
            endpoint_function = generate_remote_function(
                self.node_uid,
                signature,
                v.path,
                self.make_call,
                pre_kwargs=v.pre_kwargs,
            )
            endpoint_function.__doc__ = v.doc_string
            self._add_route(api_module, v, endpoint_function)
        self.api_module = api_module

    @property
    def services(self) -> APIModule:
        if self.api_module is None:
            self.generate_endpoints()
        return self.api_module

    def __repr__(self) -> str:
        modules = self.services
        _repr_str = "client.api.services\n"
        for attr_name in modules._modules:
            module_or_func = getattr(modules, attr_name)
            module_path_str = f"client.api.services.{attr_name}"
            _repr_str += f"\n{module_path_str}\n\n"
            if hasattr(module_or_func, "_modules"):
                for func_name in module_or_func._modules:
                    func = getattr(module_or_func, func_name)
                    sig = func.__ipython_inspector_signature_override__
                    _repr_str += f"{module_path_str}.{func_name}{sig}\n\n"
        return _repr_str


# code from here:
# https://github.com/ipython/ipython/blob/339c0d510a1f3cb2158dd8c6e7f4ac89aa4c89d8/IPython/core/oinspect.py#L370
def _render_signature(obj_signature, obj_name) -> str:
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
        rendered = "{}(\n{})".format(
            obj_name, "".join("    {},\n".format(r) for r in result)
        )
    else:
        rendered = "{}({})".format(obj_name, ", ".join(result))

    if obj_signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(obj_signature.return_annotation)
        rendered += " -> {}".format(anno)

    return rendered


def _getdef(self, obj, oname="") -> Union[str, None]:
    """Return the call signature for any callable object.
    If any exception is generated, None is returned instead and the
    exception is suppressed."""
    try:
        return _render_signature(signature(obj), oname)
    except:  # noqa: E722
        return None


def monkey_patch_getdef(self, obj, oname="") -> Union[str, None]:
    try:
        if hasattr(obj, "__ipython_inspector_signature_override__"):
            return _render_signature(
                getattr(obj, "__ipython_inspector_signature_override__"), oname
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
    # print("Failed to monkeypatch IPython Signature Override")
    pass  # nosec


@serializable()
class NodeView(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    node_name: str
    verify_key: SyftVerifyKey

    @staticmethod
    def from_api(api: SyftAPI):
        # stores the name root verify key of the domain node
        node_metadata = api.connection.get_node_metadata(api.signing_key)
        return NodeView(
            node_name=node_metadata.name,
            verify_key=SyftVerifyKey.from_string(node_metadata.verify_key),
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NodeView):
            return False
        return self.node_name == other.node_name and self.verify_key == other.verify_key

    def __hash__(self) -> int:
        return hash((self.node_name, self.verify_key))
