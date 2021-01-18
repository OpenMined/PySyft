import os
from typing import Any
from typing import Union
from typing import TextIO
from loguru import logger
from typing import NoReturn

LOG_FORMAT = "[{time}][{level}][{module}] {message}"

logger.remove()
DEFAULT_LOG_FILE = "syft_{time}.log"


def disable_logging() -> None:
    logger.remove()


def add_logger(
    file_path: Union[None, str, os.PathLike, TextIO] = None,
    log_level: str = "ERROR",
) -> None:
    log_file = DEFAULT_LOG_FILE if file_path is None else file_path
    try:
        logger.add(
            log_file,
            format=LOG_FORMAT,
            enqueue=True,
            colorize=False,
            diagnose=True,
            backtrace=True,
            rotation="10 MB",
            retention="1 day",
            level=log_level,
        )
    except BaseException:
        logger.add(
            log_file,
            format=LOG_FORMAT,
            enqueue=True,
            colorize=False,
            diagnose=True,
            backtrace=True,
            level=log_level,
        )


def traceback_and_raise(*args: Any, **kargs: Any) -> NoReturn:
    try:
        logger.opt(lazy=True).exception(*args, **kargs)
    except BaseException as e:
        logger.debug("failed to print exception", e)
    raise Exception(str(*args, **kargs))


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
