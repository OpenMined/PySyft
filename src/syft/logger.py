import os
from typing import Any
from typing import Union
from typing import TextIO
from typing import NoReturn

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
    except BaseException as e:
        logger.debug("failed to print exception", e)
    raise e


def traceback(*args: Any, **kargs: Any) -> None:
    try:
        logger.opt(lazy=True).exception(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print exception", e)


def critical(*args: Any, **kargs: Any) -> None:
    try:
        logger.opt(lazy=True).critical(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print error", e)


def error(*args: Any, **kargs: Any) -> None:
    try:
        logger.opt(lazy=True).error(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print error", e)


def warning(*args: Any, **kargs: Any) -> None:
    try:
        logger.opt(lazy=True).warning(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print warning", e)


def info(*args: Any, **kargs: Any) -> None:
    try:
        logger.opt(lazy=True).info(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print info", e)


def debug(*args: Any, **kargs: Any) -> None:
    try:
        logger.opt(lazy=True).debug(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print debug", e)


def trace(*args: Any, **kargs: Any) -> None:
    try:
        logger.opt(lazy=True).trace(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print trace", e)
