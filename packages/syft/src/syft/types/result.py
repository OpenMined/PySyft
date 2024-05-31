# stdlib
from collections.abc import Callable
import functools
from typing import Generic
from typing import Literal
from typing import NoReturn
from typing import TypeAlias
from typing import TypeVar

T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True, bound=BaseException)
BE = TypeVar("BE", bound=BaseException)


class Ok(Generic[T]):
    __slots__ = ("value",)
    __match_args__ = ("ok_value",)

    def __init__(self, value: T):
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)

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

    def unwrap(self) -> T:
        return self.value


class Error(Generic[E]):
    __slots__ = ("value",)
    __match_args__ = ("error_value",)

    def __init__(self, value: E):
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)

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

    def unwrap(self) -> NoReturn:
        if isinstance(self.value, BaseException):
            raise self.value
        raise TypeError("Error is not a BaseException")


Result: TypeAlias = Ok[T] | Error[E]


def catch(
    *exceptions: type[BE],
) -> Callable[[Callable[..., T]], Callable[..., Result[T, BE]]]:
    if not exceptions or not all(
        issubclass(exception, BaseException) for exception in exceptions
    ):
        raise TypeError("The catch() decorator only accepts exceptions")

    def decorator(func: Callable[..., T]) -> Callable[..., Result[T, BE]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Result[T, BE]:
            try:
                return Ok(func(*args, **kwargs))
            except exceptions as e:
                return Error(e)

        return wrapper

    return decorator
