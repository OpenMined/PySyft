# stdlib
import traceback

from collections.abc import Callable
import functools
from typing import Generic
from typing import Literal
from typing import NoReturn
from typing import ParamSpec
from typing import TypeAlias
from typing import TypeVar

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


class Err(Generic[E]):
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


Result: TypeAlias = Ok[T] | Err[E]


def as_result(
    *exceptions: type[BE],
) -> Callable[[Callable[P, T]], Callable[P, Result[T, BE]]]:
    if not exceptions or not all(
        issubclass(exception, BaseException) for exception in exceptions
    ):
        raise TypeError("The as_result() decorator only accepts exceptions")

    def decorator(func: Callable[P, T]) -> Callable[P, Result[T, BE]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, BE]:
            try:
                output = func(*args, **kwargs)
                if isinstance(output, Ok) or isinstance(output, Err):
                    raise TypeError((
                        f"Functions decorated with `as_result` should not return Result.\n"
                        f"function output: {output}"
                    ))
                return Ok(func(*args, **kwargs))
            except exceptions as e:
                # We want to adjust the traceback so we can remove the decorator
                # and the wrapper from the traceback for nicer error messages.
                frames = []

                for tb in traceback.walk_tb(e.__traceback__):
                    frames.append(tb)

                # The first two frames we always want to keep as they show the
                # immediate context of the error: this except block  and the caller
                frames_to_keep = frames[0:2]

                # Next, we skip two frames that contains the decorator and the wrapper.
                # They'll appear when unwrap() is called. We don't need them.
                if len(frames) > 2:
                    frames_to_keep += frames[4:]

                # Before being done, we need to adjust the traceback.tb_next so that
                # the frames are linked together properly.
                for i, tb in enumerate(frames_to_keep):
                    if i + 1 == len(frames_to_keep):
                        tb.tb_next = None
                    else:
                        tb.tb_next = frames_to_keep[i + 1]

                # Finally, we create a new exception with the adjusted traceback.
                # ex = e.with_traceback(keepers[0])
                e.__traceback__ = frames_to_keep[0]

                return Err(e)

        return wrapper

    return decorator
