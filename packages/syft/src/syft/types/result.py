# stdlib
from collections.abc import Callable
import functools
from typing import Any
from typing import Final
from typing import Generic
from typing import Literal
from typing import NoReturn
from typing import ParamSpec
from typing import TypeAlias
from typing import TypeVar

# relative
from .errors import SyftException
from .errors import exclude_from_traceback
from .errors import process_traceback

T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True, bound=BaseException)
BE = TypeVar("BE", bound=BaseException)
P = ParamSpec("P")


class Ok(Generic[T]):
    __slots__ = ("value",)
    __match_args__ = ("ok_value",)

    def __init__(self, value: T):
        self.value = value

    def __repr__(self) -> str:
        return f"Ok({self.value})"

    @property
    def ok_value(self) -> T:
        return self.value

    def err(self) -> None:
        return None

    def ok(self) -> T:
        return self.value

    def is_err(self) -> Literal[False]:
        return False

    def is_ok(self) -> Literal[True]:
        return True

    def unwrap(self, *args: Any, **kwargs: Any) -> T:
        return self.value


class Err(Generic[E]):
    __slots__ = ("value",)
    __match_args__ = ("error_value",)

    def __init__(self, value: E):
        self.value = value

    def __repr__(self) -> str:
        return f"Err({self.value})"

    @property
    def error_value(self) -> E:
        return self.value

    def err(self) -> E:
        return self.value

    def ok(self) -> None:
        return None

    def is_err(self) -> Literal[True]:
        return True

    def is_ok(self) -> Literal[False]:
        return False

    @exclude_from_traceback
    def unwrap(
        self, public_message: str | None = None, private_message: str | None = None
    ) -> NoReturn:
        if isinstance(self.value, SyftException):
            if public_message is not None:
                self.value.public_message = public_message
            if private_message is not None:
                self.value._private_message = private_message
        if isinstance(self.value, BaseException):
            raise self.value
        raise TypeError("Error is not a BaseException")


OkErr: Final = (Ok, Err)
Result: TypeAlias = Ok[T] | Err[E]


def as_result(
    *exceptions: type[BE], convert_to_syft_exception: bool = False
) -> Callable[[Callable[P, T]], Callable[P, Result[T, BE]]]:
    if not exceptions or not all(
        issubclass(exception, BaseException) for exception in exceptions
    ):
        raise TypeError("The as_result() decorator only accepts exceptions")

    class _AsResultError(Exception): ...

    def decorator(func: Callable[P, T]) -> Callable[P, Result[T, BE]]:
        @exclude_from_traceback
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, BE]:
            try:
                output = func(*args, **kwargs)
                if isinstance(output, Ok) or isinstance(output, Err):
                    raise _AsResultError(
                        f"Functions decorated with `as_result` should not return Result.\n"
                        f"Did you forget to unwrap() the result in {func.__name__}?\n"
                        f"result: {output}"
                    )
                return Ok(output)
            except exceptions as exc:
                if convert_to_syft_exception and not isinstance(exc, SyftException):
                    exc = SyftException.from_exception(exc)  # type: ignore
                exc = process_traceback(exc)
                return Err(exc)

        return wrapper

    return decorator
