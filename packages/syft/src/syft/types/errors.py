# stdlib
from collections.abc import Callable
from importlib import import_module
import inspect
import os
import traceback
from types import CodeType
from types import TracebackType
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
import uuid

# third party
from IPython import get_ipython
from IPython.display import HTML
from IPython.display import display
import psutil
from typing_extensions import Self

# relative
from ..service.user.user_roles import ServiceRole
from ..util.notebook_ui.components.tabulator_template import jinja_env

if TYPE_CHECKING:
    # relative
    from ..service.context import AuthedServiceContext


class SyftException(Exception):
    """
    A Syft custom exception class with distinct public and private messages.

    Attributes:
        private_message (str): Detailed error message intended for administrators.
        public_message (str): General error message for end-users.
    """

    public_message = "An error occurred. Contact the admin for more information."

    def __init__(
        self,
        private_message: str | None = None,
        public_message: str | None = None,
        server_trace: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if public_message is not None and not isinstance(public_message, str):
            raise TypeError("public message should be a string")
        if private_message is not None and not isinstance(private_message, str):
            raise TypeError("private message should be a string")

        if public_message:
            self.public_message = public_message

        self._private_message = private_message or ""
        self._server_trace = server_trace or ""
        super().__init__(self.public, *args, **kwargs)

    @property
    def public(self) -> str:
        """
        Returns the public error message.

        Returns:
            str: The public error message.
        """
        return self.public_message

    def get_message(self, context: "AuthedServiceContext") -> str:
        """
        Retrieves the appropriate message based on the user's role, obtained via
        `context.role`.

        Args:
            context (AuthedServiceContext): The server context.

        Returns:
            str: The private or public message based on the role.
        """
        if context.role.value >= ServiceRole.DATA_OWNER.value or context.dev_mode:
            return self._private_message or self.public
        return self.public

    def get_tb(
        self,
        context: "AuthedServiceContext | None" = None,
        overwrite_permission: bool = False,
    ) -> str | None:
        """
        Returns the error traceback as a string, if the user is able to see it.

        Args:
            context (AuthedServiceContext): The authenticated service context which
                contains the user's role.

        Returns:
            str | None: A string representation of the current stack trace if the
                user is a DataOwner or higher, otherwise None.
        """
        # stdlib
        import traceback

        if (
            overwrite_permission
            or (context and context.role.value >= ServiceRole.DATA_OWNER.value)
            or (context and context.dev_mode)
        ):
            return "".join(traceback.format_exception(self))
        return None

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        public_message: str | None = None,
        private_message: str | None = None,
    ) -> Self:
        """
        Builds an instance of SyftException from an existing exception. It allows
        setting a public message for end users or resetting the private message.
        If no private_message is provided, the original exception's message is used.
        If no public_message is provided, the default public message is used.

        Args:
            exception (BaseException): The original exception from which to create
                the new instance. The message from this exception will be used as
                the base message for the new instance.

            public_message (str, optional): An optional message intended for public
                display. This message can provide user-friendly information about
                the error. If not provided, the default message is used.

            private_message (str, optional): An optional message intended for private
                display. This message should provide more information about the error
                to administrators. If not provided, the exception's message is used.

        Returns:
            Self: A new instance of the class. The new instance retains the traceback
                of the original exception.
        """
        if isinstance(exc, SyftException):
            private_message = private_message or exc._private_message
            public_message = public_message or exc.public_message
        elif isinstance(exc, BaseException):
            private_message = private_message or str(exc)

        new_exc = cls(private_message, public_message=public_message)
        new_exc.__traceback__ = exc.__traceback__
        new_exc.__cause__ = exc
        new_exc = process_traceback(new_exc)
        return new_exc

    @property
    def _repr_html_class_(self) -> str:
        return "alert-danger"

    def __str__(self) -> str:
        # this assumes that we show the server side error on the client side without a jupyter notebook
        server_trace = self._server_trace
        message = self._private_message or self.public

        if server_trace:
            message = f"{message}\nserver_trace: {server_trace}"

        return message

    def _repr_html_(self) -> str:
        is_dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
        display = "block" if self._server_trace or is_dev_mode else "none"

        exc = process_traceback(self)
        _traceback_str_list = traceback.format_exception(exc)
        traceback_str = "".join(_traceback_str_list)

        table_template = jinja_env.get_template("syft_exception.jinja2")
        table_html = table_template.render(
            name=type(self).__name__,
            html_id=uuid.uuid4().hex,
            server_trace=self._server_trace,
            message=self._private_message or self.public,
            traceback_str=traceback_str,
            display=display,
            dev_mode=is_dev_mode,
        )
        return table_html


class raises:
    def __init__(self, expected_exception, show=False):  # type: ignore
        self.expected_exception = expected_exception
        self.show = show

    def __enter__(self):  # type: ignore
        # Before block of code
        pass

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        message = None
        expected_exception_type = self.expected_exception
        if not isinstance(expected_exception_type, type):
            expected_exception_type = type(self.expected_exception)
            if hasattr(self.expected_exception, "public_message"):
                message = self.expected_exception.public_message.replace("*", "")

        # After block of code
        if exc_type is None:
            raise AssertionError(
                f"Expected {self.expected_exception} to be raised, "
                "but no exception was raised."
            )
        if not issubclass(exc_type, expected_exception_type):
            raise AssertionError(
                f"Expected {expected_exception_type} to be raised, but got {exc_type}."
            )
        if message and message not in exc_value.public_message:
            raise AssertionError(
                f"Expected {expected_exception_type} to be raised, "
                f"did not contain {message}."
            )
        if self.show:
            # keep this print!
            print("with sy.raises successfully caught the following exception:")
            if hasattr(exc_value, "_repr_html_"):
                display(HTML(exc_value._repr_html_()))
            else:
                print(
                    f"The following exception was catched\n{exc_value}",
                )
        return True  # Suppress the exception


class ExceptionFilter(tuple):
    """
    Filter and store all exception classes from a given module path. This class can be
    used in try/except blocks to handle these exceptions as a group (see example).

    Attributes:
        module (str): The name of the module from which exceptions are filtered.

    Example:
        ```
        from syft.types.errors import ExceptionFilter

        try:
            ...
        except ExceptionFilter("google.cloud.bigquery") as e:
            ...
        ```
    """

    def __init__(self, module: str) -> None:
        self.module = module

    def __new__(cls, module: str) -> Self:
        """
        Creates a new instance of ExceptionFilter, which gathers all exception classes
        from the specified module and stores them as a tuple.
        """
        exceptions: tuple[type[BaseException], ...]

        try:
            imported_module = import_module(module)
        except ModuleNotFoundError:
            # TODO: log warning
            exceptions = ()
        else:
            exceptions = tuple(
                obj
                for _, obj in inspect.getmembers(imported_module, inspect.isclass)
                if issubclass(obj, BaseException) and not issubclass(obj, Warning)
            )

        instance = super().__new__(cls, exceptions)

        instance.module = module

        return instance


_excluded_code_objects: set[CodeType] = set()

E = TypeVar("E", bound=BaseException)
F = TypeVar("F", bound=Callable[..., object])


def exclude_from_traceback(f: F) -> F:
    """
    Decorator to mark a function to be removed from the traceback when an
    exception is raised. This is useful for functions that are not relevant to
    the error message and would only clutter the traceback.
    """
    _excluded_code_objects.add(f.__code__)
    return f


def process_traceback(exc: E) -> E:
    """
    Adjusts the traceback of an exception to remove specific frames related to
    the as_result decorator and the unwrap() call, for cleaner and more relevant
    error messages.

    Args:
        exc (E): The exception whose traceback is to be adjusted.

    Returns:
        E: The same exception with an adjusted traceback.
    """
    # We want to adjust the traceback so we can remove frames which contain
    # a function marked with the _excluded_code_objects decorator, improving
    # the stacktrace of the error messages.
    tb = exc.__traceback__
    frames: list[TracebackType] = []

    while tb is not None:
        if tb.tb_frame.f_code not in _excluded_code_objects:
            frames.append(tb)
        tb = tb.tb_next

    # Before being done, we need to adjust the traceback.tb_next so that
    # the frames are linked together properly.
    for i, tb in enumerate(frames):
        if i + 1 == len(frames):
            tb.tb_next = None
        else:
            tb.tb_next = frames[i + 1]

    return exc


class CredentialsError(SyftException):
    public_message = "Invalid credentials."


def syft_exception_handler(
    shell: Any, etype: Any, evalue: Any, tb: Any, tb_offset: Any = None
) -> None:
    display(HTML(evalue._repr_html_()))


runs_in_pytest = False
for pid in psutil.pids():
    try:
        if "PYTEST_CURRENT_TEST" in psutil.Process(pid).environ():
            runs_in_pytest = True
    except Exception:
        pass  # nosec


# be very careful when changing this. pytest (with nbmake) will
# not pick up exceptions if they have a custom exception handler (fail silently)
if not runs_in_pytest:
    try:
        get_ipython().set_custom_exc((SyftException,), syft_exception_handler)  # noqa: F821
    except Exception:
        pass  # nosec
