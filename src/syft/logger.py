# stdlib
import os
from typing import Any
from typing import Callable
from typing import NoReturn
from typing import TextIO
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
        try:
            method = getattr(logger.opt(lazy=True), level, None)
            if "print" in kwargs and kwargs["print"] is True:
                del kwargs["print"]
                print(*args, **kwargs)
            if method is not None:
                method(*args, **kwargs)
            else:
                raise Exception(f"no method {level} on logger")
        except BaseException as e:
            logger.debug("failed to log exception", e)

    return log_and_print


def traceback(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="exception")(*args, **kwargs)


def critical(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="critical")(*args, **kwargs)


def error(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="error")(*args, **kwargs)


def warning(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="warning")(*args, **kwargs)


def info(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="info")(*args, **kwargs)


def debug(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="debug")(*args, **kwargs)


def trace(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="trace")(*args, **kwargs)
