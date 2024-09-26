# stdlib
import ast
from collections.abc import Callable
import inspect
from inspect import Signature
import keyword
import linecache
import re
import textwrap
from textwrap import dedent
from typing import Any
from typing import cast

# third party
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator

# relative
from ...abstract_server import AbstractServer
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...serde.signature import signature_remove_context
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_migration import migrate
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.transforms import TransformContext
from ...types.transforms import drop
from ...types.transforms import generate_action_object_id
from ...types.transforms import generate_id
from ...types.transforms import keep
from ...types.transforms import make_set_default
from ...types.transforms import transform
from ...types.uid import UID
from ...util.misc_objs import MarkdownDescription
from ..context import AuthedServiceContext
from ..response import SyftError
from ..user.user import UserView
from .utils import print as log_print

NOT_ACCESSIBLE_STRING = "N / A"


class HelperFunctionSet:
    def __init__(self, helper_functions: dict[str, Callable]) -> None:
        self.helper_functions = helper_functions
        for name, func in helper_functions.items():
            setattr(self, name, func)


class TwinAPIAuthedContext(AuthedServiceContext):
    __canonical_name__ = "TwinAPIAuthedContext"
    __version__ = SYFT_OBJECT_VERSION_1

    user: UserView | None = None
    settings: dict[str, Any] | None = None
    code: HelperFunctionSet | None = None
    state: dict[Any, Any] | None = None
    admin_client: SyftClient | None = None
    user_client: SyftClient | None = None


@serializable()
class TwinAPIContextView(SyftObject):
    __canonical_name__ = "TwinAPIContextView"
    __version__ = SYFT_OBJECT_VERSION_1

    __repr_attrs__ = ["settings", "state", "user"]
    user: UserView
    settings: dict[str, Any]
    state: dict[Any, Any]


def get_signature(func: Callable) -> Signature:
    sig = inspect.signature(func)
    sig = signature_remove_context(sig)
    return sig


def register_fn_in_linecache(fname: str, src: str) -> None:
    """adds a function to linecache, such that inspect.getsource works for functions nested in this function.
    This only works if the same function is compiled under the same filename"""
    lines = [
        line + "\n" for line in src.splitlines()
    ]  # use same splitting method same as linecache 112 (py3.12)
    linecache.cache[fname] = (137, None, lines, fname)


@serializable()
class TwinAPIEndpointViewV1(SyftObject):
    # version
    __canonical_name__ = "CustomAPIView"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    action_object_id: UID
    signature: Signature
    access: str = "Public"
    mock_function: str | None = None
    private_function: str | None = None
    description: MarkdownDescription | None = None
    mock_helper_functions: list[str] | None = None
    private_helper_functions: list[str] | None = None
    worker_pool: str | None = None
    endpoint_timeout: int = 60


@serializable()
class TwinAPIEndpointView(SyftObject):
    # version
    __canonical_name__ = "CustomAPIView"
    __version__ = SYFT_OBJECT_VERSION_2

    path: str
    action_object_id: UID
    signature: Signature
    access: str = "Public"
    mock_function: str | None = None
    private_function: str | None = None
    description: MarkdownDescription | None = None
    mock_helper_functions: list[str] | None = None
    private_helper_functions: list[str] | None = None
    worker_pool_name: str | None = None
    endpoint_timeout: int = 60

    __repr_attrs__ = [
        "path",
        "signature",
        "worker_pool_name",
        "endpoint_timeout",
    ]

    def _coll_repr_(self) -> dict[str, Any]:
        if self.mock_function:
            mock_parsed_code = ast.parse(self.mock_function)
            mock_function_name = [
                server.name
                for server in ast.walk(mock_parsed_code)
                if isinstance(server, ast.FunctionDef)
            ][0]
        else:
            mock_function_name = NOT_ACCESSIBLE_STRING

        if self.private_function:
            private_parsed_code = ast.parse(self.private_function)
            private_function_name = [
                server.name
                for server in ast.walk(private_parsed_code)
                if isinstance(server, ast.FunctionDef)
            ][0]
        else:
            private_function_name = NOT_ACCESSIBLE_STRING

        worker_pool_name = "UNSET (DEFAULT)"
        if self.worker_pool_name is not None:
            worker_pool_name = self.worker_pool_name
        return {
            "API path": self.path,
            "Signature": self.path + str(self.signature),
            "Access": self.access,
            "Mock Function": mock_function_name,
            "Private Function": private_function_name,
            "Worker Pool": worker_pool_name,
        }


@serializable()
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

    __exclude_sync_diff_attrs__ = ["state"]

    def __repr__(self) -> str:
        type_name = type(self).__name__
        repr_str = f"""<{type_name}: {self.func_name}>

        {self.api_code}
        """
        return textwrap.dedent(repr_str)

    @field_validator("api_code", check_fields=False)
    @classmethod
    def validate_api_code(cls, api_code: str) -> str:
        valid_code = True
        api_code = dedent(api_code)
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
        if not str.isidentifier(func_name) or keyword.iskeyword(func_name):
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

    def build_internal_context(
        self,
        context: AuthedServiceContext,
        admin_client: SyftClient | None = None,
        user_client: SyftClient | None = None,
    ) -> TwinAPIAuthedContext:
        helper_function_dict: dict[str, Callable] = {}
        self.helper_functions = self.helper_functions or {}
        for helper_name, helper_code in self.helper_functions.items():
            # Create a dictionary to serve as local scope
            local_scope: dict[str, Callable] = {}

            # Execute the function string within the local scope
            exec(helper_code, local_scope)  # nosec
            helper_function_dict[helper_name] = local_scope[helper_name]

        helper_function_set = HelperFunctionSet(helper_function_dict)

        user = context.server.services.user.get_current_user(context)

        return TwinAPIAuthedContext(
            credentials=context.credentials,
            role=context.role,
            job_id=context.job_id,
            extra_kwargs=context.extra_kwargs,
            has_execute_permissions=context.has_execute_permissions,
            server=context.server,
            id=context.id,
            settings=self.settings or {},
            code=helper_function_set,
            state=self.state or {},
            user=user,
            admin_client=admin_client,
            user_client=user_client,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # relative
        from ... import SyftSigningKey
        from ..context import AuthedServiceContext

        mock_context = AuthedServiceContext(
            server=AbstractServer(), credentials=SyftSigningKey.generate().verify_key
        )
        return self.call_locally(mock_context, *args, **kwargs)

    def call_locally(
        self, context: AuthedServiceContext, *args: Any, **kwargs: Any
    ) -> Any:
        inner_function = ast.parse(self.api_code).body[0]
        inner_function.decorator_list = []
        # compile the function
        raw_byte_code = compile(ast.unparse(inner_function), "<string>", "exec")

        # load it
        exec(raw_byte_code)  # nosec

        internal_context = self.build_internal_context(context=context)

        # execute it
        evil_string = f"{self.func_name}(*args, **kwargs,context=internal_context)"
        result = eval(evil_string, None, locals())  # nosec

        # Update code context state
        self.update_state(internal_context.state)

        # return the results
        return result


@serializable()
class PrivateAPIEndpoint(Endpoint):
    # version
    __canonical_name__ = "PrivateAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    view_access: bool = False


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

        if path.startswith(".") or path.endswith("."):
            raise ValueError("Path cannot start or end with a '.'")
        if not path.islower():
            raise ValueError("Path must be lowercase")
        parts = path.split(".")
        for part in parts:
            if not str.isidentifier(part) or keyword.iskeyword(part):
                raise ValueError(f"Invalid path: {part} is not a valid identifier")

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
    description: MarkdownDescription | None = None
    endpoint_timeout: int = 60


@serializable()
class CreateTwinAPIEndpointV1(BaseTwinAPIEndpoint):
    # version
    __canonical_name__ = "CreateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    private_function: PrivateAPIEndpoint | None = None
    mock_function: PublicAPIEndpoint
    signature: Signature
    description: MarkdownDescription | None = None
    worker_pool: str | None = None
    endpoint_timeout: int = 60


@serializable()
class CreateTwinAPIEndpoint(BaseTwinAPIEndpoint):
    # version
    __canonical_name__ = "CreateTwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_2

    path: str
    private_function: PrivateAPIEndpoint | None = None
    mock_function: PublicAPIEndpoint
    signature: Signature
    description: MarkdownDescription | None = None
    worker_pool_name: str | None = None
    endpoint_timeout: int = 60

    def __init__(
        self, description: str | MarkdownDescription | None = "", **kwargs: Any
    ) -> None:
        if isinstance(description, str):
            description = MarkdownDescription(text=description)

        super().__init__(**kwargs, description=description)


@serializable()
class TwinAPIEndpointV1(SyncableSyftObject):
    # version
    __canonical_name__: str = "TwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1
    __exclude_sync_diff_attrs__ = ["private_function"]
    __private_sync_attr_mocks__ = {
        "private_function": None,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    path: str
    private_function: PrivateAPIEndpoint | None = None
    mock_function: PublicAPIEndpoint
    signature: Signature
    description: MarkdownDescription | None = None
    action_object_id: UID
    worker_pool: str | None = None
    endpoint_timeout: int = 60

    __attr_searchable__ = ["path"]
    __attr_unique__ = ["path"]
    __repr_attrs__ = [
        "path",
        "description",
        "private_function",
        "mock_function",
        "endpoint_timeout",
    ]


@serializable()
class TwinAPIEndpoint(SyncableSyftObject):
    # version
    __canonical_name__: str = "TwinAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_2
    __exclude_sync_diff_attrs__ = ["private_function"]
    __private_sync_attr_mocks__ = {
        "private_function": None,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    path: str
    private_function: PrivateAPIEndpoint | None = None
    mock_function: PublicAPIEndpoint
    signature: Signature
    description: MarkdownDescription | None = None
    action_object_id: UID
    worker_pool_name: str | None = None
    endpoint_timeout: int = 60

    __attr_searchable__ = ["path"]
    __attr_unique__ = ["path"]
    __repr_attrs__ = [
        "path",
        "description",
        "private_function",
        "mock_function",
        "endpoint_timeout",
    ]

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

    def select_code(
        self, context: AuthedServiceContext
    ) -> PrivateAPIEndpoint | PublicAPIEndpoint | None:
        """Select the code to execute based on the user's permissions and public code availability.

        Args:
            context: The context of the user requesting the code.
        Returns:
            Result[Ok, Err]: The selected code to execute.
        """
        if self.has_permission(context) and self.private_function:
            return self.private_function
        return self.mock_function

    def exec(
        self,
        context: AuthedServiceContext,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the code based on the user's permissions and public code availability.

        Args:
            context: The context of the user requesting the code.
            *args: Any
            **kwargs: Any
        Returns:
            Any: The result of the executed code.
        """
        selected_code = self.select_code(context)
        return self.exec_code(selected_code, context, *args, log_id=log_id, **kwargs)

    def exec_mock_function(
        self,
        context: AuthedServiceContext,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the public code if it exists."""
        if self.mock_function:
            return self.exec_code(
                self.mock_function, context, *args, log_id=log_id, **kwargs
            )

        raise SyftException(public_message="No public code available")

    def exec_private_function(
        self,
        context: AuthedServiceContext,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the private code if user is has the proper permissions.

        Args:
            context: The context of the user requesting the code.
            *args: Any
            **kwargs: Any
        Returns:
            Any: The result of the executed code.
        """
        if self.private_function is None:
            raise SyftException(public_message="No private code available")

        if self.has_permission(context):
            return self.exec_code(
                self.private_function, context, *args, log_id=log_id, **kwargs
            )

        raise SyftException(public_message="You're not allowed to run this code.")

    def get_user_client_from_server(self, context: AuthedServiceContext) -> SyftClient:
        # get a user client
        guest_client = context.server.get_guest_client()
        user_client = guest_client
        private_key = context.server.services.user.signing_key_for_verify_key(
            context.credentials
        )
        signing_key = private_key.signing_key
        user_client.credentials = signing_key
        return user_client

    def get_admin_client_from_server(self, context: AuthedServiceContext) -> SyftClient:
        admin_client = context.server.get_guest_client()
        admin_client.credentials = context.server.signing_key
        return admin_client

    @as_result(SyftException)
    def exec_code(
        self,
        code: PrivateAPIEndpoint | PublicAPIEndpoint,
        context: AuthedServiceContext,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> Any:
        # stdlib
        import builtins as __builtin__
        import functools

        original_print = __builtin__.print

        try:
            if log_id is not None:
                print = functools.partial(log_print, context, log_id)
            else:
                print = original_print  # type: ignore

            inner_function = ast.parse(code.api_code).body[0]
            inner_function.decorator_list = []
            # compile the function
            src = ast.unparse(inner_function)
            raw_byte_code = compile(src, code.func_name, "exec")
            register_fn_in_linecache(code.func_name, src)
            user_client = self.get_user_client_from_server(context)
            admin_client = self.get_admin_client_from_server(context)

            internal_context = code.build_internal_context(
                context=context, admin_client=admin_client, user_client=user_client
            )
            evil_string = f"{code.func_name}(*args, **kwargs,context=internal_context)"

            _globals = {"print": print}
            # load it
            exec(raw_byte_code, _globals, locals())  # nosec

            # execute it
            evil_string = f"{code.func_name}(*args, **kwargs,context=internal_context)"
            result = None
            try:
                # users can raise SyftException in their code
                result = eval(evil_string, _globals, locals())  # nosec
            except SyftException as e:
                # capture it as the result variable
                result = e

            # run all this code to clean up the state
            code.update_state(internal_context.state)

            if isinstance(code, PublicAPIEndpoint):
                self.mock_function = code
            else:
                self.private_function = code  # type: ignore

            api_service = context.server.get_service("apiservice")
            api_service.stash.upsert(
                context.server.services.user.root_verify_key, self
            ).unwrap()

            print = original_print  # type: ignore
            # if we caught a SyftException above we will raise and auto wrap to Result
            if isinstance(result, SyftException):
                raise result

            # here we got a non Exception result which will also be wrapped in Result
            # return the results
            return result
        except Exception as e:
            # If it's admin, return the error message.
            # TODO: cleanup typeerrors
            if context.role.value == 128 or isinstance(e, TypeError):
                raise SyftException(
                    public_message=f"An error was raised during the execution of the API endpoint call: \n {str(e)}"
                )
            else:
                raise SyftException(
                    public_message=(
                        "Oops something went wrong during this endpoint execution, "
                        "please contact your admin."
                    )
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
    decorator_regex = r"@.*?def "

    # Substituting the matched pattern with "def"
    return re.sub(decorator_regex, "def ", code, count=1, flags=re.DOTALL)


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

            context.server = cast(AbstractServer, context.server)
            admin_key = context.server.services.user.root_verify_key

            # If endpoint exists **AND** (has visible access **OR** the user is admin)
            if endpoint_type is not None and (
                endpoint_type.view_access or context.credentials == admin_key
            ):
                context.output[code_field] = decorator_cleanup(endpoint_type.api_code)
                context.output[helper_function_field] = (
                    endpoint_type.helper_functions.values() or []
                )
            else:
                context.output[code_field] = None
                context.output[helper_function_field] = []
        return context

    return code_string


@transform(TwinAPIAuthedContext, TwinAPIContextView)
def twin_api_context_to_twin_api_context_view() -> list[Callable]:
    return [keep(["state", "settings", "user"])]


@transform(CreateTwinAPIEndpoint, TwinAPIEndpoint)
def endpoint_create_to_twin_endpoint() -> list[Callable]:
    return [generate_id, generate_action_object_id, check_and_cleanup_signature]


@transform(TwinAPIEndpoint, TwinAPIEndpointView)
def twin_endpoint_to_view() -> list[Callable]:
    return [
        set_access_type,
        extract_code_string("private_function"),
        extract_code_string("mock_function"),
    ]


@transform(Endpoint, PrivateAPIEndpoint)
def endpoint_to_private_endpoint() -> list[Callable]:
    return [
        keep(
            [
                "api_code",
                "func_name",
                "settings",
                "view_access",
                "helper_functions",
                "state",
                "signature",
            ]
        )
    ]


@transform(Endpoint, PublicAPIEndpoint)
def endpoint_to_public_endpoint() -> list[Callable]:
    return [
        keep(
            [
                "api_code",
                "func_name",
                "settings",
                "view_access",
                "helper_functions",
                "state",
                "signature",
            ]
        )
    ]


@migrate(TwinAPIEndpointV1, TwinAPIEndpoint)
def migrate_twin_api_endpoint_v1_to_current() -> list[Callable]:
    return [
        drop(["worker_pool"]),
        make_set_default("worker_pool_name", None),
    ]


@migrate(CreateTwinAPIEndpointV1, CreateTwinAPIEndpoint)
def migrate_create_twin_api_endpoint_v1_to_current() -> list[Callable]:
    return [
        drop(["worker_pool"]),
        make_set_default("worker_pool_name", None),
    ]


@migrate(TwinAPIEndpointViewV1, TwinAPIEndpointView)
def migrate_twin_api_endpoint_view_v1_to_current() -> list[Callable]:
    return [
        drop(["worker_pool"]),
        make_set_default("worker_pool_name", None),
    ]


@migrate(TwinAPIEndpointView, TwinAPIEndpointViewV1)
def migrate_twin_api_endpoint_view_current_to_v1() -> list[Callable]:
    return [
        drop(["worker_pool_name"]),
        make_set_default("worker_pool", None),
    ]


@migrate(CreateTwinAPIEndpoint, CreateTwinAPIEndpointV1)
def migrate_create_twin_api_endpoint_current_to_v1() -> list[Callable]:
    return [
        drop(["worker_pool_name"]),
        make_set_default("worker_pool", None),
    ]


@migrate(TwinAPIEndpoint, TwinAPIEndpointV1)
def migrate_twin_api_endpoint_current_to_v1() -> list[Callable]:
    return [
        drop(["worker_pool_name"]),
        make_set_default("worker_pool", None),
    ]


def api_endpoint(
    path: str,
    settings: dict[str, str] | None = None,
    helper_functions: list[Callable] | None = None,
    description: MarkdownDescription | None = None,
    worker_pool_name: str | None = None,
    endpoint_timeout: int = 60,
) -> Callable[..., TwinAPIEndpoint | SyftError]:
    def decorator(f: Callable) -> TwinAPIEndpoint | SyftError:
        try:
            helper_functions_dict = {
                f.__name__: dedent(inspect.getsource(f))
                for f in (helper_functions or [])
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
                worker_pool_name=worker_pool_name,
                endpoint_timeout=endpoint_timeout,
            )
        except ValidationError as e:
            for error in e.errors():
                error_msg = error["msg"]
            res = SyftError(message=error_msg)
        return res

    return decorator


def api_endpoint_method(
    settings: dict[str, str] | None = None,
    helper_functions: list[Callable] | None = None,
) -> Callable[..., Endpoint | SyftError]:
    def decorator(f: Callable) -> Endpoint | SyftError:
        try:
            helper_functions_dict = {
                f.__name__: dedent(inspect.getsource(f))
                for f in (helper_functions or [])
            }
            return Endpoint(
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
    mock_function: Endpoint,
    private_function: Endpoint | None = None,
    description: MarkdownDescription | None = None,
    worker_pool_name: str | None = None,
    endpoint_timeout: int = 60,
    hide_mock_definition: bool = False,
    hide_private_definition: bool = True,
) -> CreateTwinAPIEndpoint | SyftError:
    try:
        # Parse the string to extract the function name

        endpoint_signature = mock_function.signature
        if private_function is not None:
            if private_function.signature != mock_function.signature:
                return SyftError(message="Signatures don't match")
            endpoint_signature = mock_function.signature
            private_function.view_access = not hide_private_definition
            mock_function.view_access = not hide_mock_definition

            return CreateTwinAPIEndpoint(
                path=path,
                private_function=private_function.to(PrivateAPIEndpoint),
                mock_function=mock_function.to(PublicAPIEndpoint),
                signature=endpoint_signature,
                description=description,
                worker_pool_name=worker_pool_name,
                endpoint_timeout=endpoint_timeout,
            )

        return CreateTwinAPIEndpoint(
            path=path,
            prublic_code=mock_function.to(PublicAPIEndpoint),
            signature=endpoint_signature,
            worker_pool_name=worker_pool_name,
            endpoint_timeout=endpoint_timeout,
        )
    except ValidationError as e:
        for error in e.errors():
            error_msg = error["msg"]

    return SyftError(message=error_msg)
