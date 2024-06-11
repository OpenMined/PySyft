# stdlib
import sys

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
                if isinstance(output, Result):
                    raise TypeError((
                        f"Functions decorated with `as_result` should not return Result.\n"
                        f"function output: {output}"
                    ))
                return Ok(func(*args, **kwargs))
            except exceptions as e:
                # exc_traceback = e.__traceback__
                # tb1 = e.__traceback__ #1
                # # This skips two frames: the frame of the decorator itself
                # # and the frame of the wrapper
                # try:
                #     tb2 = tb1.tb_next #2
                #     tb3 = tb2.tb_next #3
                #     tb4 = tb3.tb_next #4
                #     tb5 = tb4.tb_next #4

                #     tb3.tb_next = tb5
                #     tb1.tb_next = tb3
                # except Exception:
                #     pass
                # ex = e.with_traceback(tb1)
                # levels = e.__traceback__
                tb = e.__traceback__
                levels = []
                while tb is not None:
                    levels.append(tb)
                    tb = tb.tb_next

                keepers = levels[0:2]
                if len(levels) > 2:
                    keepers += levels[4:]

                print(f"Len of keepers: {len(keepers)}")
                print(f"Len of levels: {len(levels)}")

                tb = e.__traceback__
                while tb is not None:
                    levels.append(tb)
                    tb = tb.tb_next


                for index, keeper in enumerate(keepers):
                    print(f"{index} in keeper == {levels.index(keeper)} in levels")

                import traceback
                for level in levels:
                    print(f"{traceback.format_tb(level)[0]}")

                for i, tb in enumerate(keepers):
                    print(f"index: {i}")
                    if i + 1 == len(keepers):
                        tb.tb_next = None
                    else:
                        tb.tb_next = keepers[i + 1]

                ex = e.with_traceback(keepers[0])

                return Err(ex)

        return wrapper

    return decorator
