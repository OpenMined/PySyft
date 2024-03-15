# stdlib
import ast
from collections.abc import Callable
import inspect
from inspect import Signature
import keyword
import re
from typing import Any

# third party
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...serde.signature import signature_remove_context
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
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
    access: str = "Public"

    __repr_attrs__ = [
        "path",
        "signature",
    ]

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "API path": self.path,
            "Signature": self.path + str(self.signature),
            "Access": self.access,
        }


class Endpoint(SyftObject):
    """Base class to perform basic Endpoint validation for both public/private endpoints."""

    # version
    __canonical_name__ = "CustomApiEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    @field_validator("api_code", check_fields=False)
    @classmethod
    def validate_api_code(cls, api_code: str) -> str:
        valid_code = True
        try:
            ast.parse(api_code)
        except SyntaxError:
            # If the code isn't valid Python syntax
            valid_code = False

        if not valid_code:
            raise ValueError("Code must be a valid Python function.")

        return api_code

    @field_validator("func_name", check_fields=False)
    @classmethod
    def validate_func_name(cls, func_name: str) -> str:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", func_name) or keyword.iskeyword(
            func_name
        ):
            raise ValueError("Invalid function name.")
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
        if not re.match(r"^[a-z]+(\.[a-z]+)*$", path):
            raise ValueError('String must be a path-like string (e.g., "new.endpoint")')
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    path: str
    private_code: PrivateAPIEndpoint
    public_code: PublicAPIEndpoint | None = None
    signature: Signature

    __attr_searchable__ = ["path"]
    __attr_unique__ = ["path"]

    def has_mock(self) -> bool:
        return self.api_mock_code is not None

    def has_permission(self, context: AuthedServiceContext) -> bool:
        """Check if the user has permission to access the endpoint.

        Args:
            context: The context of the user requesting the code.
        Returns:
            bool: True if the user has permission to access the endpoint, False otherwise.
        """
        if context.role.value == 128:
            return True
        return False

    def select_code(self, context: AuthedServiceContext) -> Result[Ok, Err]:
        """Select the code to execute based on the user's permissions and public code availability.

        Args:
            context: The context of the user requesting the code.
        Returns:
            Result[Ok, Err]: The selected code to execute.
        """
        if self.has_permission(context):
            return Ok(self.private_code)

        if self.public_code:
            return Ok(self.public_code)

        return Err("No public code available")

    def exec(self, context: AuthedServiceContext, *args: Any, **kwargs: Any) -> Any:
        """Execute the code based on the user's permissions and public code availability.

        Args:
            context: The context of the user requesting the code.
            *args: Any
            **kwargs: Any
        Returns:
            Any: The result of the executed code.
        """
        result = self.select_code(context)
        if result.is_err():
            return context, SyftError(message=result.err())

        selected_code = result.ok()
        return self.exec_code(selected_code, context, *args, **kwargs)

    def exec_public_code(
        self, context: AuthedServiceContext, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the public code if it exists."""
        if self.public_code:
            return self.exec_code(self.public_code, context, *args, **kwargs)
        return context, SyftError(message="No public code available")

    def exec_private_code(
        self, context: AuthedServiceContext, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the private code if user is has the proper permissions.

        Args:
            context: The context of the user requesting the code.
            *args: Any
            **kwargs: Any
        Returns:
            Any: The result of the executed code.
        """
        if self.has_permission(context):
            return self.exec_code(self.private_code, context, *args, **kwargs)

        return context, SyftError(message="You're not allowed to run this code.")

    def exec_code(
        self, code: str, context: AuthedServiceContext, *args: Any, **kwargs: Any
    ) -> Any:
        try:
            inner_function = ast.parse(code.api_code).body[0]
            inner_function.decorator_list = []
            # compile the function
            raw_byte_code = compile(ast.unparse(inner_function), "<string>", "exec")
            # load it
            exec(raw_byte_code)  # nosec
            # execute it
            evil_string = f"{code.func_name}(context, *args, **kwargs)"
            result = eval(evil_string, None, locals())  # nosec
            # return the results
            return context, result
        except Exception as e:
            print(f"Failed to run CustomAPIEndpoint Code. {e}")
            return SyftError(message=e)


def set_access_type(context: TransformContext) -> TransformContext:
    if context.output is not None and context.obj is not None:
        if context.obj.public_code is not None:
            context.output["access"] = "Public"
        else:
            context.output["access"] = "Private"
    return context


@transform(CreateTwinAPIEndpoint, TwinAPIEndpoint)
def endpoint_create_to_twin_endpoint() -> list[Callable]:
    return [generate_id]


@transform(TwinAPIEndpoint, TwinAPIEndpointView)
def twin_endpoint_to_view() -> list[Callable]:
    return [
        set_access_type,
        drop("private_code"),
        drop("public_code"),
    ]


def api_endpoint(path: str) -> Callable[..., TwinAPIEndpoint | SyftError]:
    def decorator(f: Callable) -> TwinAPIEndpoint | SyftError:
        try:
            res = CreateTwinAPIEndpoint(
                path=path,
                private_code=PrivateAPIEndpoint(
                    api_code=inspect.getsource(f),
                    func_name=f.__name__,
                ),
                signature=get_signature(f),
            )
        except ValidationError as e:
            for error in e.errors():
                error_msg = error["msg"]
            res = SyftError(message=error_msg)
        return res

    return decorator


def create_new_api_endpoint(
    path: str,
    private: Callable[..., Any],
    description: str | None = None,
    public: Callable[..., Any] | None = None,
    private_configs: dict[str, Any] | None = None,
    public_configs: dict[str, Any] | None = None,
) -> CreateTwinAPIEndpoint | SyftError:
    try:
        if public is not None:
            return CreateTwinAPIEndpoint(
                path=path,
                private_code=PrivateAPIEndpoint(
                    api_code=inspect.getsource(private),
                    func_name=private.__name__,
                    context_vars=private_configs,
                ),
                public_code=PublicAPIEndpoint(
                    api_code=inspect.getsource(public),
                    func_name=public.__name__,
                    context_vars=public_configs,
                ),
                signature=get_signature(private),
            )

        return CreateTwinAPIEndpoint(
            path=path,
            private_code=PrivateAPIEndpoint(
                api_code=inspect.getsource(private),
                func_name=private.__name__,
                context_vars=private_configs,
            ),
            signature=get_signature(private),
        )
    except ValidationError as e:
        for error in e.errors():
            error_msg = error["msg"]

    return SyftError(message=error_msg)
