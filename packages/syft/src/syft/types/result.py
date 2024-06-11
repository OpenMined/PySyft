# stdlib
from collections.abc import Callable
import functools
from types import TracebackType
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
                    raise TypeError(
                        f"Functions decorated with `as_result` should not return Result.\n"
                        f"output: {output}"
                    )
                return Ok(func(*args, **kwargs))
            except exceptions as exc:
                strip_asresult_from_exception_traceback(exc)
                return Err(exc)

        return wrapper

    return decorator


def strip_asresult_from_exception_traceback(exc: BaseException) -> BaseException:
    """
    Adjusts the traceback of an exception to remove specific frames related to
    the as_result decorator and the unwrap() call, for cleaner and more relevant
    error messages.

    This function modifies the traceback of the provided exception by keeping the first
    two frames (which show the immediate context of the error) and then skipping the
    next two frames (which contain decorator and wrapper frames). The rest of the frames
    are preserved.

    Args:
        exc (BaseException): The exception whose traceback is to be adjusted.

    Returns:
        BaseException: The same exception with an adjusted traceback.
    """
    # We want to adjust the traceback so we can remove the decorator
    # and the wrapper from the traceback for nicer error messages.
    tb = exc.__traceback__
    frames: list[TracebackType] = []

    while tb is not None:
        frames.append(tb)
        tb = tb.tb_next

    # The first two frames we always want to keep as they show the immediate
    # context of the error: the current except block and the caller
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

    return exc
