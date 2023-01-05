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

# third party
from nacl.exceptions import BadSignatureError
import requests
from result import Err
from result import Ok
from result import OkErr
from result import Result
from typeguard import check_type

# relative
from ....core.common.serde.recursive import index_syft_by_module_name
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.deserialize import _deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize
from ...common.uid import UID
from .action_service import ActionService
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .signature import Signature
from .user import UserCollection


class APIRegistry:
    __api_registry__: Dict[str, SyftAPI] = {}

    @classmethod
    def set_api_for(cls, node_uid: UID, api: SyftAPI) -> None:
        cls.__api_registry__[node_uid] = api

    @classmethod
    def api_for(cls, node_uid: UID) -> SyftAPI:
        return cls.__api_registry__[node_uid]


@serializable(recursive_serde=True)
class APIEndpoint(SyftObject):
    path: str
    name: str
    description: str
    doc_string: str
    signature: Signature
    has_self: bool = False


def signature_remove_self(signature: Signature) -> Signature:
    params = dict(signature.parameters)
    params.pop("self")
    return Signature(
        list(params.values()), return_annotation=signature.return_annotation
    )


def signature_remove_credentials(signature: Signature) -> Signature:
    params = dict(signature.parameters)
    params.pop("credentials")
    return Signature(
        list(params.values()), return_annotation=signature.return_annotation
    )


@serializable(recursive_serde=True)
class SignedSyftAPICall(SyftObject):
    __canonical_name__ = "SyftAPICall"
    __version__ = 1

    __attr_allowlist__ = ["signature", "credentials", "serialized_message"]
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
    def is_valid(self) -> Result[bool, Err]:
        try:
            _ = self.credentials.verify_key.verify(
                self.serialized_message, self.signature
            )
        except BadSignatureError:
            return Err("BadSignatureError")

        return Ok(True)


@serializable(recursive_serde=True)
class SyftAPICall(SyftObject):
    # version
    __canonical_name__ = "SyftAPICall"
    __version__ = 1
    __attr_allowlist__ = ["path", "args", "kwargs"]

    # fields
    path: str
    args: List[SyftObject]
    kwargs: Dict[str, Any]

    # serde / storage rules
    __attr_state__ = ["path", "args", "kwargs"]

    def sign(self, credentials: SyftSigningKey) -> SignedSyftAPICall:
        signed_message = credentials.signing_key.sign(_serialize(self, to_bytes=True))

        return SignedSyftAPICall(
            credentials=credentials.verify_key,
            serialized_message=signed_message.message,
            signature=signed_message.signature,
        )


def generate_remote_function(signature: Signature, path: str, make_call: Callable):
    def wrapper(*args, **kwargs):
        # need real Signature object
        # params = signature.bind(*args, **kwargs)
        if len(kwargs) == 0:
            # 🟡 TODO 15: Rewrite wrapper API functions to handle, args and kwargs properly
            raise Exception("Please use kwargs")
        for key, value in kwargs.items():
            if key not in signature.parameters:
                raise Exception("Wrong key", key, "for sig", signature)
            param = signature.parameters[key]
            if isinstance(param.annotation, str):
                t = index_syft_by_module_name(param.annotation)
            else:
                t = param.annotation

            msg = None
            try:
                check_type(key, value, t)
            except TypeError:
                msg = f"{key} must be {t.__name__} not {type(value).__name__}"

            if msg:
                raise Exception(msg)

        api_call = SyftAPICall(path=path, args=[], kwargs=kwargs)
        result = make_call(api_call=api_call)
        return result

    wrapper.__ipython_inspector_signature_override__ = signature
    return wrapper


class APIModule:
    pass


@serializable(recursive_serde=True)
class SyftAPI(SyftObject):
    # version
    __canonical_name__ = "SyftAPI"
    __version__ = 1
    __attr_allowlist__ = ["endpoints"]

    # fields
    node_uid: Optional[UID] = None
    endpoints: Dict[str, APIEndpoint]
    api_module: Optional[APIModule] = None
    api_url: str = ""
    signing_key: Optional[SyftSigningKey] = None
    # serde / storage rules
    __attr_state__ = ["endpoints"]

    def __post_init__(self) -> None:
        # 🟡 TODO 16: Write user login and key retrieval / local caching
        self.signing_key = SyftSigningKey.generate()

    @staticmethod
    def for_user(node_uid: UID) -> SyftAPI:
        # 🟡 TODO 1: Filter SyftAPI with User VerifyKey
        # relative

        endpoints = {
            # 🟡 TODO 2: Change endpoint keys to use . syntax and build a tree of modules
            "services_user_create": APIEndpoint(
                path="services.user.create",
                name="create",
                description="Create User",
                doc_string=UserCollection.create.__doc__,
                signature=signature(UserCollection.create),
                has_self=False,
            ),
            "action_store_send": APIEndpoint(
                path="services.action.set",
                name="send",
                description="Send Action Objects",
                doc_string=ActionService.set.__doc__,
                signature=signature(ActionService.set),
                has_self=False,
            ),
            "action_store_get": APIEndpoint(
                path="services.action.get",
                name="get",
                description="Get Action Objects",
                doc_string=ActionService.get.__doc__,
                signature=signature(ActionService.get),
                has_self=False,
            ),
            "action_store_execute": APIEndpoint(
                path="services.action.execute",
                name="execute",
                description="Execute Actions",
                doc_string=ActionService.execute.__doc__,
                signature=signature(ActionService.execute),
                has_self=False,
            ),
        }
        return SyftAPI(node_uid=node_uid, endpoints=endpoints)

    def make_call(self, api_call: SyftAPICall) -> None:
        signed_call = api_call.sign(credentials=self.signing_key)
        msg_bytes: bytes = _serialize(obj=signed_call, to_bytes=True)
        response = requests.post(
            url=self.api_url,
            data=msg_bytes,
        )

        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch metadata. Response returned with code {response.status_code}"
            )

        result = _deserialize(response.content, from_bytes=True)
        if isinstance(result, OkErr):
            if result.is_ok():
                return result.ok()
            else:
                return result.err()
        return result

    def generate_endpoints(self) -> None:
        api_module = APIModule()
        for k, v in self.endpoints.items():
            signature = v.signature
            if not v.has_self:
                signature = signature_remove_self(signature)
            signature = signature_remove_credentials(signature)
            endpoint_function = generate_remote_function(
                signature, v.path, self.make_call
            )
            endpoint_function.__doc__ = v.doc_string
            setattr(api_module, k, endpoint_function)
        self.api_module = api_module

    @property
    def services(self) -> APIModule:
        if self.api_module is None:
            self.generate_endpoints()
        return self.api_module


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
    print("Failed to monkeypatch IPython Signature Override")
    pass
