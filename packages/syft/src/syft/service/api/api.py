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
from ...types.transforms import generate_id
from ...types.transforms import transform
from ..context import AuthedServiceContext
from ..response import SyftError


def signature_remove_secrets(signature: Signature) -> Signature:
    params = dict(signature.parameters)
    params.pop("secrets", None)
    return Signature(
        list(params.values()), return_annotation=signature.return_annotation
    )


def get_signature(func: Callable) -> Signature:
    sig = inspect.signature(func)
    sig = signature_remove_context(sig)
    sig = signature_remove_secrets(sig)
    return sig


@serializable()
class TwinAPIEndpointView(SyftObject):
    # version
    __canonical_name__ = "CustomAPIView"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    signature: Signature
    access: str = "Public"
    public_code: str | None = None
    private_code: str | None = None

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

    @field_validator("secrets", check_fields=False)
    @classmethod
    def validate_secrets(cls, secrets: dict[str, Any] | None) -> dict[str, Any] | None:
        return secrets


@serializable()
class PrivateAPIEndpoint(Endpoint):
    # version
    __canonical_name__ = "PrivateAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    api_code: str
    func_name: str
    settings: dict[str, Any] | None = None
    view_access: bool = True


@serializable()
class PublicAPIEndpoint(Endpoint):
    # version
    __canonical_name__ = "PublicAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    api_code: str
    func_name: str
    secrets: dict[str, Any] | None = None
    view_access: bool = True


class BaseTwinAPIEndpoint(SyftObject):
    __canonical_name__ = "BaseTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

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

    @field_validator("path", check_fields=False)
    @classmethod
    def validate_path(cls, path: str) -> str:
        if not re.match(r"^[a-z]+(\.[a-z]+)*$", path):
            raise ValueError('String must be a path-like string (e.g., "new.endpoint")')
        return path

    @field_validator("private_code", check_fields=False)
    @classmethod
    def validate_private_code(
        cls, private_code: PrivateAPIEndpoint
    ) -> PrivateAPIEndpoint:
        return private_code

    @field_validator("public_code", check_fields=False)
    @classmethod
    def validate_public_code(
        cls, public_code: PublicAPIEndpoint | None
    ) -> PublicAPIEndpoint | None:
        return public_code


@serializable()
class UpdateTwinAPIEndpoint(PartialSyftObject, BaseTwinAPIEndpoint):
    # version
    __canonical_name__ = "UpdateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_code: PrivateAPIEndpoint
    public_code: PublicAPIEndpoint


@serializable()
class CreateTwinAPIEndpoint(BaseTwinAPIEndpoint):
    # version
    __canonical_name__ = "CreateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_code: PrivateAPIEndpoint
    public_code: PublicAPIEndpoint | None = None
    signature: Signature


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
        self,
        code: PrivateAPIEndpoint | PublicAPIEndpoint,
        context: AuthedServiceContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            inner_function = ast.parse(code.api_code).body[0]
            inner_function.decorator_list = []
            # compile the function
            raw_byte_code = compile(ast.unparse(inner_function), "<string>", "exec")
            # load it
            exec(raw_byte_code)  # nosec
            # execute it
            if code.secrets is None:
                code.secrets = {}
            evil_string = f"{code.func_name}(*args, **kwargs, secrets=code.secrets,context=context)"
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


def check_and_cleanup_signature(context: TransformContext) -> TransformContext:
    if context.output is not None and context.obj is not None:
        params = dict(context.obj.signature.parameters)
        if "secrets" not in params or "context" not in params:
            raise ValueError(
                "Function Signature must include 'secrets' (Dict[str,str]) and 'context' [AuthedContext] parameters."
            )
        params.pop("secrets", None)
        params.pop("context", None)
        context.output["signature"] = Signature(
            list(params.values()),
            return_annotation=context.obj.signature.return_annotation,
        )
    return context


def decorator_cleanup(code: str) -> str:
    # Regular expression to remove decorator
    # It matches from "@" to "def" (non-greedy) across multiple lines
    decorator_regex = r"@.*?def"

    # Substituting the matched pattern with "def"
    return re.sub(decorator_regex, "def", code, count=1, flags=re.DOTALL)


def extract_code_string(code_field: str) -> Callable:
    def code_string(context: TransformContext) -> TransformContext:
        if context.obj is not None and context.output is not None:
            endpoint_type = (
                context.obj.private_code
                if code_field == "private_code"
                else context.obj.public_code
            )

            if endpoint_type is not None and endpoint_type.view_access:
                context.output[code_field] = decorator_cleanup(endpoint_type.api_code)
            else:
                context.output[code_field] = "N / A"
        return context

    return code_string


@transform(CreateTwinAPIEndpoint, TwinAPIEndpoint)
def endpoint_create_to_twin_endpoint() -> list[Callable]:
    return [generate_id, check_and_cleanup_signature]


@transform(TwinAPIEndpoint, TwinAPIEndpointView)
def twin_endpoint_to_view() -> list[Callable]:
    return [
        set_access_type,
        extract_code_string("private_code"),
        extract_code_string("public_code"),
    ]


def api_endpoint(
    path: str, secrets: dict[str, str] | None = None
) -> Callable[..., TwinAPIEndpoint | SyftError]:
    def decorator(f: Callable) -> TwinAPIEndpoint | SyftError:
        try:
            res = CreateTwinAPIEndpoint(
                path=path,
                private_code=PrivateAPIEndpoint(
                    api_code=inspect.getsource(f),
                    func_name=f.__name__,
                    secrets=secrets,
                ),
                signature=inspect.signature(f),
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
    private_secrets: dict[str, Any] | None = None,
    public_secrets: dict[str, Any] | None = None,
) -> CreateTwinAPIEndpoint | SyftError:
    try:
        if public is not None:
            return CreateTwinAPIEndpoint(
                path=path,
                private_code=PrivateAPIEndpoint(
                    api_code=inspect.getsource(private),
                    func_name=private.__name__,
                    secrets=private_secrets,
                ),
                public_code=PublicAPIEndpoint(
                    api_code=inspect.getsource(public),
                    func_name=public.__name__,
                    secrets=public_secrets,
                ),
                signature=inspect.signature(private),
            )

        return CreateTwinAPIEndpoint(
            path=path,
            private_code=PrivateAPIEndpoint(
                api_code=inspect.getsource(private),
                func_name=private.__name__,
                secrets=private_secrets,
            ),
            signature=inspect.signature(private),
        )
    except ValidationError as e:
        for error in e.errors():
            error_msg = error["msg"]

    return SyftError(message=error_msg)
