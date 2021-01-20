# stdlib
import os
from typing import Any
from typing import NoReturn
from typing import TextIO
from typing import Callable
from typing import Union

# third party
from loguru import logger

LOG_FORMAT = "[{time}][{level}][{module}] {message}"

logger.remove()
DEFAULT_SINK = "syft_{time}.log"


def remove() -> None:
    logger.remove()


def add(
    sink: Union[None, str, os.PathLike, TextIO] = None,
    level: str = "ERROR",
) -> None:
    sink = DEFAULT_SINK if sink is None else sink
    try:
        logger.add(
            sink=sink,
            format=LOG_FORMAT,
            enqueue=True,
            colorize=False,
            diagnose=True,
            backtrace=True,
            rotation="10 MB",
            retention="1 day",
            level=level,
        )
    except BaseException:
        logger.add(
            sink=sink,
            format=LOG_FORMAT,
            enqueue=True,
            colorize=False,
            diagnose=True,
            backtrace=True,
            level=level,
        )


def traceback_and_raise(e: Any, verbose: bool = False) -> NoReturn:
    try:
        if verbose:
            logger.opt(lazy=True).exception(e)
        else:
            logger.opt(lazy=True).critical(e)
    except BaseException as ex:
        logger.debug("failed to print exception", ex)
    raise e


def create_log_and_print_function(level: str) -> Callable:
    def log_and_print(*args: Any, **kwargs: Any) -> None:
        attr_name = level
        try:
            if level == "traceback":
                attr_name = "exception"  # this one is different

            method = getattr(logger.opt(lazy=True), attr_name, None)
            if "print" in kwargs and kwargs["print"] is True:
                del kwargs["print"]
                print(*args, **kwargs)
            if method is not None:
                method(*args, **kwargs)
            else:
                raise Exception(f"no method {attr_name} on logger")
        except BaseException as e:
            logger.debug("failed to log exception", e)

    return log_and_print


log_function_names = [
    "traceback",
    "critical",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
]
log_functions = {}
for func in log_function_names:
    log_functions[func] = create_log_and_print_function(level=func)


# when importing the dynamically generated functions in log_function_names this
# will return the correct function
def __getattr__(name: str) -> Callable:
    if name in log_function_names:
        return log_functions[name]
    else:
        return super.__getattr__(name)  # type: ignore


__all__ = ["remove", "add", "traceback_and_raise"] + log_function_names
