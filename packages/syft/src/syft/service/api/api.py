# stdlib
import ast
from collections.abc import Callable
import inspect
from inspect import Signature
from typing import Any

# third party
from pydantic import field_validator
from pydantic import model_validator

# relative
from ...serde.serializable import serializable
from ...serde.signature import signature_remove_context
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import generate_id
from ...types.transforms import transform
from ..context import AuthedServiceContext
from ..response import SyftError


def get_signature(func: Callable) -> Signature:
    sig = inspect.signature(func)
    sig = signature_remove_context(sig)
    return sig


@serializable()
class TwinAPIEndpointView(SyftObject):
    # version
    __canonical_name__ = "CustomAPIView"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    signature: Signature

    __repr_attrs__ = [
        "path",
        "signature",
    ]

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "API path": self.path,
            "Signature": self.path + str(self.signature),
        }


class Endpoint(SyftObject):
    """Base class to perform basic Endpoint validation for both public/private endpoints."""

    # version
    __canonical_name__ = "CustomApiEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    @field_validator("api_code", check_fields=False)
    @classmethod
    def validate_api_code(cls, api_code: str) -> str:
        return api_code

    @field_validator("func_name", check_fields=False)
    @classmethod
    def validate_func_name(cls, func_name: str) -> str:
        return func_name

    @field_validator("context_vars", check_fields=False)
    @classmethod
    def validate_context_vars(
        cls, context_vars: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return context_vars


@serializable()
class PrivateAPIEndpoint(Endpoint):
    # version
    __canonical_name__ = "PrivateAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    api_code: str
    func_name: str
    context_vars: dict[str, Any] | None = None


@serializable()
class PublicAPIEndpoint(Endpoint):
    # version
    __canonical_name__ = "PublicAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    api_code: str
    func_name: str
    context_vars: dict[str, Any] | None = None


@serializable()
class UpdateTwinAPIEndpoint(PartialSyftObject):
    # version
    __canonical_name__ = "UpdateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_code: PrivateAPIEndpoint
    public_code: PublicAPIEndpoint


@serializable()
class CreateTwinAPIEndpoint(SyftObject):
    # version
    __canonical_name__ = "CreateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_code: PrivateAPIEndpoint
    public_code: PublicAPIEndpoint | None = None
    signature: Signature

    @model_validator(mode="before")
    @classmethod
    def validate_signature(cls, data: dict[str, Any]) -> dict[str, Any]:
        # TODO: Implement a signature check.
        mismatch_signatures = False
        if data.get("public_code") is not None and mismatch_signatures:
            raise ValueError(
                "Public and Private API Endpoints must have the same signature."
            )

        return data

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        if path == "":
            raise ValueError("path cannot be empty")
        return path

    @field_validator("private_code")
    @classmethod
    def validate_private_code(
        cls, private_code: PrivateAPIEndpoint
    ) -> PrivateAPIEndpoint:
        return private_code

    @field_validator("public_code")
    @classmethod
    def validate_public_code(
        cls, public_code: PublicAPIEndpoint | None
    ) -> PublicAPIEndpoint | None:
        return public_code


@serializable()
class TwinAPIEndpoint(SyftObject):
    # version
    __canonical_name__ = "TwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_code: PrivateAPIEndpoint
    public_code: PublicAPIEndpoint | None = None
    signature: Signature

    __attr_searchable__ = ["path"]
    __attr_unique__ = ["path"]

    def has_mock(self) -> bool:
        return self.api_mock_code is not None

    def exec(self, context: AuthedServiceContext, **kwargs: Any) -> Any:
        try:
            inner_function = ast.parse(self.api_code).body[0]
            inner_function.decorator_list = []
            # compile the function
            raw_byte_code = compile(ast.unparse(inner_function), "<string>", "exec")
            # load it
            exec(raw_byte_code)  # nosec
            # execute it
            evil_string = f"{self.func_name}(context, **kwargs)"
            result = eval(evil_string, None, locals())  # nosec
            # return the results
            return context, result
        except Exception as e:
            print(f"Failed to run CustomAPIEndpoint Code. {e}")
            return SyftError(message=e)


@transform(CreateTwinAPIEndpoint, TwinAPIEndpoint)
def endpoint_create_to_twin_endpoint() -> list[Callable]:
    return [generate_id]


@transform(TwinAPIEndpoint, TwinAPIEndpointView)
def twin_endpoint_to_view() -> list[Callable]:
    return [drop("private_code"), drop("public_code")]


def api_endpoint(path: str) -> Callable[..., TwinAPIEndpoint]:
    def decorator(f: Callable) -> TwinAPIEndpoint:
        res = CreateTwinAPIEndpoint(
            path=path,
            private_code=PrivateAPIEndpoint(
                api_code=inspect.getsource(f),
                func_name=f.__name__,
            ),
            signature=get_signature(f),
        )
        return res

    return decorator
