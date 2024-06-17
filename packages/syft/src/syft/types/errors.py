# stdlib
from collections.abc import Callable
from importlib import import_module
import inspect
from types import CodeType
from types import TracebackType
from typing import Any
from typing import TypeVar

# third party
from typing_extensions import Self

# relative
from ..service.context import AuthedServiceContext
from ..service.user.user_roles import ServiceRole


class SyftException(Exception):
    """
    A Syft custom exception class with distinct public and private messages.

    Attributes:
        private_message (str): Detailed error message intended for administrators.
        public_message (str): General error message for end-users.
    """

    public_message = "An error occurred. Contact the admin for more information."

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        exc = super().__new__(cls, *args, **kwargs)
        # removes irrelevant frames from the traceback (e.g. as_result decorator)
        process_traceback(exc)
        return exc

    def __init__(
        self,
        private_message: str | None = None,
        public_message: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if public_message:
            self.public_message = public_message
        self._private_message = private_message
        super().__init__(self.public, *args, **kwargs)

    @property
    def public(self) -> str:
        """
        Returns the public error message.

        Returns:
            str: The public error message.
        """
        return self.public_message

    def get_message(self, context: AuthedServiceContext) -> str:
        """
        Retrieves the appropriate message based on the user's role, obtained via
        `context.role`.

        Args:
            context (AuthedServiceContext): The context containing user role information.

        Returns:
            str: The private or public message based on the role.
        """
        if context.role.value >= ServiceRole.DATA_OWNER.value:
            return self._private_message
        return self.public

    @classmethod
    def from_exception(
        cls, exc: BaseException, public_message: str | None = None
    ) -> Self:
        """
        Builds an instance of SyftException from an existing exception, incorporating
        the message from the original exception as a private message. It also allows
        setting a public message for end users.

        Args:
            exception (BaseException): The original exception from which to create
                the new instance. The message from this exception will be used as
                the base message for the new instance.

            public_message (str, optional): An optional message intended for public
                display. This message can provide user-friendly information about
                the error. If not provided, the default is None.

        Returns:
            Self: A new instance of the class. The new instance retains the traceback
                of the original exception.
        """
        new_exc = cls(str(exc), public_message=public_message)
        new_exc.__traceback__ = exc.__traceback__
        new_exc = process_traceback(new_exc)
        return new_exc


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

    def __new__(cls, module_name: str) -> Self:
        """
        Creates a new instance of ExceptionFilter, which gathers all exception classes
        from the specified module and stores them as a tuple.
        """
        module = import_module(module_name)

        exceptions = (
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, BaseException) and not issubclass(obj, Warning)
        )

        instance = super().__new__(cls, exceptions)

        instance.module = module_name

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
