# future
from __future__ import annotations

# stdlib
import inspect
from inspect import signature
import types
from typing import Dict
from typing import Optional
from typing import Union

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from .signature import Signature


@serializable(recursive_serde=True)
class APIEndpoint(SyftObject):
    path: str
    name: str
    description: str
    doc_string: str
    signature: Signature  # TODO replace with real signature


def generate_remote_function(signature: Signature):
    class Wrapper(object):
        def __call__(self, *args, **kwargs):
            print("got signature", signature)
            # need real Signature object
            params = signature.bind(*args, **kwargs)
            print("params", params)
            print("args, kwargs", args, kwargs)
            # TODO: add validation based on signature and input types
            # TODO: make message to send Action to service with data

    return Wrapper()


class APIModule:
    pass


@serializable(recursive_serde=True)
class SyftAPI(SyftObject):
    # version
    __canonical_name__ = "SyftAPI"
    __version__ = 1
    __attr_allowlist__ = ["endpoints"]

    # fields
    endpoints: Dict[str, APIEndpoint]
    api_module: Optional[APIModule] = None
    # serde / storage rules
    __attr_state__ = ["endpoints"]

    @staticmethod
    def for_user() -> SyftAPI:
        # TODO: get user key and filter
        # relative
        from .user import UserCollection

        sig = signature(UserCollection.create)

        endpoints = {
            "services_user_create": APIEndpoint(  # TODO change to . syntax and build a tree of modules
                path="services.user.create",
                name="create",
                description="Create User",
                doc_string=UserCollection.create.__doc__,
                signature=sig,
            )
        }
        return SyftAPI(endpoints=endpoints)

    def generate_endpoints(self) -> None:
        api_module = APIModule()
        for k, v in self.endpoints.items():
            endpoint_function = generate_remote_function(v.signature)
            endpoint_function.__doc__ = v.doc_string
            endpoint_function.__ipython_inspector_signature_override__ = v.signature
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
