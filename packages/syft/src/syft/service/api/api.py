# stdlib
import ast
from collections.abc import Callable
import inspect
from inspect import Signature
import keyword
import re
from typing import Any
from typing import cast

# third party
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_node import AbstractNode
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

NOT_ACCESSIBLE_STRING = "N / A"


class HelperFunctionSet:
    def __init__(self, helper_functions: dict[str, Callable]) -> None:
        self.helper_functions = helper_functions
        for name, func in helper_functions.items():
            setattr(self, name, func)


class TwinAPIAuthedContext(AuthedServiceContext):
    __canonical_name__ = "AuthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    settings: dict[str, Any] | None = None
    code: HelperFunctionSet | None = None
    state: dict[Any, Any] | None = None


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
    mock_function: str | None = None
    private_function: str | None = None
    description: str | None = None
    mock_helper_functions: list[str] | None = None
    private_helper_functions: list[str] | None = None

    __repr_attrs__ = [
        "path",
        "signature",
    ]

    def _coll_repr_(self) -> dict[str, Any]:
        mock_parsed_code = ast.parse(self.mock_function)
        mock_function_name = [
            node.name
            for node in ast.walk(mock_parsed_code)
            if isinstance(node, ast.FunctionDef)
        ][0]
        private_function_name = NOT_ACCESSIBLE_STRING
        if self.private_function != NOT_ACCESSIBLE_STRING:
            private_parsed_code = ast.parse(self.private_function)
            private_function_name = [
                node.name
                for node in ast.walk(private_parsed_code)
                if isinstance(node, ast.FunctionDef)
            ][0]
        return {
            "API path": self.path,
            "Signature": self.path + str(self.signature),
            "Access": self.access,
            "Mock Function": mock_function_name,
            "Private Function": private_function_name,
        }


class Endpoint(SyftObject):
    """Base class to perform basic Endpoint validation for both public/private endpoints."""

    # version
    __canonical_name__ = "CustomApiEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    api_code: str
    func_name: str
    settings: dict[str, Any] | None = None
    view_access: bool = True
    helper_functions: dict[str, str] | None = None
    state: dict[Any, Any] | None = None
    signature: Signature

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

    @field_validator("settings", check_fields=False)
    @classmethod
    def validate_settings(
        cls, settings: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return settings

    def update_state(self, state: dict[Any, Any]) -> None:
        self.state = state


@serializable()
class PrivateAPIEndpoint(Endpoint):
    # version
    __canonical_name__ = "PrivateAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1


@serializable()
class PublicAPIEndpoint(Endpoint):
    # version
    __canonical_name__ = "PublicAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1


class BaseTwinAPIEndpoint(SyftObject):
    __canonical_name__ = "BaseTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    @model_validator(mode="before")
    @classmethod
    def validate_signature(cls, data: dict[str, Any]) -> dict[str, Any]:
        mock_function = data["mock_function"]  # mock_function can't be None
        private_function = data.get("private_function")

        # Add none check
        if private_function and private_function.signature != mock_function.signature:
            raise ValueError(
                "Mock and Private API Endpoints must have the same signature."
            )

        return data

    @field_validator("path", check_fields=False)
    @classmethod
    def validate_path(cls, path: str) -> str:
        # TODO: Check path doesn't collide with system endpoints

        if not re.match(r"^[a-z]+(\.[a-z]+)*$", path):
            raise ValueError('String must be a path-like string (e.g., "new.endpoint")')

        return path

    @field_validator("private_function", check_fields=False)
    @classmethod
    def validate_private_function(
        cls, private_function: PrivateAPIEndpoint | None
    ) -> PrivateAPIEndpoint | None:
        # TODO: what kind of validation should we do here?

        return private_function

    @field_validator("mock_function", check_fields=False)
    @classmethod
    def validate_mock_function(
        cls, mock_function: PublicAPIEndpoint
    ) -> PublicAPIEndpoint:
        # TODO: what kind of validation should we do here?
        return mock_function


@serializable()
class UpdateTwinAPIEndpoint(PartialSyftObject, BaseTwinAPIEndpoint):
    # version
    __canonical_name__ = "UpdateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_function: PrivateAPIEndpoint | None = None
    mock_function: PublicAPIEndpoint
    description: str | None = None


@serializable()
class CreateTwinAPIEndpoint(BaseTwinAPIEndpoint):
    # version
    __canonical_name__ = "CreateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_function: PrivateAPIEndpoint | None = None
    mock_function: PublicAPIEndpoint
    signature: Signature
    description: str | None = None


@serializable()
class TwinAPIEndpoint(SyftObject):
    # version
    __canonical_name__ = "TwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    path: str
    private_function: PrivateAPIEndpoint | None = None
    mock_function: PublicAPIEndpoint
    signature: Signature
    description: str | None = None

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
        if self.has_permission(context) and self.private_function:
            return Ok(self.private_function)
        return Ok(self.mock_function)

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
            return SyftError(message=result.err())

        selected_code = result.ok()
        return self.exec_code(selected_code, context, *args, **kwargs)

    def exec_mock_function(
        self, context: AuthedServiceContext, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the public code if it exists."""
        if self.mock_function:
            return self.exec_code(self.mock_function, context, *args, **kwargs)

        return SyftError(message="No public code available")

    def exec_private_function(
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
            return self.exec_code(self.private_function, context, *args, **kwargs)

        return SyftError(message="You're not allowed to run this code.")

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

            helper_function_dict: dict[str, Callable] = {}
            code.helper_functions = code.helper_functions or {}
            for helper_name, helper_code in code.helper_functions.items():
                # Create a dictionary to serve as local scope
                local_scope: dict[str, Callable] = {}

                # Execute the function string within the local scope
                exec(helper_code, local_scope)  # nosec
                helper_function_dict[helper_name] = local_scope[helper_name]

            helper_function_set = HelperFunctionSet(helper_function_dict)

            # load it
            exec(raw_byte_code)  # nosec

            internal_context = TwinAPIAuthedContext(
                credentials=context.credentials,
                role=context.role,
                job_id=context.job_id,
                extra_kwargs=context.extra_kwargs,
                has_execute_permissions=context.has_execute_permissions,
                node=context.node,
                id=context.id,
                settings=code.settings or {},
                code=helper_function_set,
                state=code.state or {},
            )
            # execute it
            evil_string = f"{code.func_name}(*args, **kwargs,context=internal_context)"
            result = eval(evil_string, None, locals())  # nosec

            # Update code context state
            code.update_state(internal_context.state)

            if isinstance(code, PublicAPIEndpoint):
                self.mock_function = code
            else:
                self.private_function = code  # type: ignore

            api_service = context.node.get_service("apiservice")
            upsert_result = api_service.stash.upsert(
                context.node.get_service("userservice").admin_verify_key(), self
            )

            if upsert_result.is_err():
                raise Exception(upsert_result.err())

            # return the results
            return result
        except Exception as e:
            # If it's admin, return the error message.
            if context.role.value == 128:
                return SyftError(message=f"{str(e)}")
            else:
                return SyftError(
                    message="Ops something went wrong during this endpoint execution, please contact your admin."
                )


def set_access_type(context: TransformContext) -> TransformContext:
    if context.output is not None and context.obj is not None:
        if context.obj.private_function is not None:
            context.output["access"] = "Private / Mock"
        else:
            context.output["access"] = "Public"
    return context


def check_and_cleanup_signature(context: TransformContext) -> TransformContext:
    if context.output is not None and context.obj is not None:
        params = dict(context.obj.signature.parameters)
        if "context" not in params:
            raise ValueError(
                "Function Signature must include 'context' [AuthedContext] parameters."
            )
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
                context.obj.private_function
                if code_field == "private_function"
                else context.obj.mock_function
            )
            helper_function_field = (
                "mock_helper_functions"
                if code_field == "mock_function"
                else "private_helper_functions"
            )

            context.node = cast(AbstractNode, context.node)
            admin_key = context.node.get_service("userservice").admin_verify_key()

            # If endpoint exists **AND** (has visible access **OR** the user is admin)
            if endpoint_type is not None and (
                endpoint_type.view_access or context.credentials == admin_key
            ):
                context.output[code_field] = decorator_cleanup(endpoint_type.api_code)
                context.output[helper_function_field] = (
                    endpoint_type.helper_functions.values() or []
                )
            else:
                context.output[code_field] = NOT_ACCESSIBLE_STRING
                context.output[helper_function_field] = []
        return context

    return code_string


@transform(CreateTwinAPIEndpoint, TwinAPIEndpoint)
def endpoint_create_to_twin_endpoint() -> list[Callable]:
    return [generate_id, check_and_cleanup_signature]


@transform(TwinAPIEndpoint, TwinAPIEndpointView)
def twin_endpoint_to_view() -> list[Callable]:
    return [
        set_access_type,
        extract_code_string("private_function"),
        extract_code_string("mock_function"),
    ]


def api_endpoint(
    path: str,
    settings: dict[str, str] | None = None,
    helper_functions: list[Callable] | None = None,
    description: str | None = None,
) -> Callable[..., TwinAPIEndpoint | SyftError]:
    def decorator(f: Callable) -> TwinAPIEndpoint | SyftError:
        try:
            helper_functions_dict = {
                f.__name__: inspect.getsource(f) for f in (helper_functions or [])
            }
            res = CreateTwinAPIEndpoint(
                path=path,
                mock_function=PublicAPIEndpoint(
                    api_code=inspect.getsource(f),
                    func_name=f.__name__,
                    settings=settings,
                    signature=inspect.signature(f),
                    helper_functions=helper_functions_dict,
                ),
                signature=inspect.signature(f),
                description=description,
            )
        except ValidationError as e:
            for error in e.errors():
                error_msg = error["msg"]
            res = SyftError(message=error_msg)
        return res

    return decorator


def private_api_endpoint(
    settings: dict[str, str] | None = None,
    helper_functions: list[Callable] | None = None,
) -> Callable[..., PrivateAPIEndpoint | SyftError]:
    def decorator(f: Callable) -> PrivateAPIEndpoint | SyftError:
        try:
            helper_functions_dict = {
                f.__name__: inspect.getsource(f) for f in (helper_functions or [])
            }
            return PrivateAPIEndpoint(
                api_code=inspect.getsource(f),
                func_name=f.__name__,
                settings=settings,
                signature=inspect.signature(f),
                helper_functions=helper_functions_dict,
            )
        except ValidationError as e:
            for error in e.errors():
                error_msg = error["msg"]
            res = SyftError(message=error_msg)
        return res

    return decorator


def mock_api_endpoint(
    settings: dict[str, str] | None = None,
    helper_functions: list[Callable] | None = None,
) -> Callable[..., PublicAPIEndpoint | SyftError]:
    def decorator(f: Callable) -> PublicAPIEndpoint | SyftError:
        try:
            helper_functions_dict = {
                f.__name__: inspect.getsource(f) for f in (helper_functions or [])
            }
            return PublicAPIEndpoint(
                api_code=inspect.getsource(f),
                func_name=f.__name__,
                settings=settings,
                signature=inspect.signature(f),
                helper_functions=helper_functions_dict,
            )
        except ValidationError as e:
            for error in e.errors():
                error_msg = error["msg"]
            res = SyftError(message=error_msg)
        return res

    return decorator


def create_new_api_endpoint(
    path: str,
    mock_function: PublicAPIEndpoint,
    private_function: PrivateAPIEndpoint | None = None,
    description: str | None = None,
) -> CreateTwinAPIEndpoint | SyftError:
    try:
        # Parse the string to extract the function name

        endpoint_signature = mock_function.signature
        if private_function is not None:
            if private_function.signature != mock_function.signature:
                return SyftError(message="Signatures don't match")
            endpoint_signature = mock_function.signature

            return CreateTwinAPIEndpoint(
                path=path,
                private_function=private_function,
                mock_function=mock_function,
                signature=endpoint_signature,
                description=description,
            )

        return CreateTwinAPIEndpoint(
            path=path,
            prublic_code=mock_function,
            signature=endpoint_signature,
        )
    except ValidationError as e:
        for error in e.errors():
            error_msg = error["msg"]

    return SyftError(message=error_msg)
