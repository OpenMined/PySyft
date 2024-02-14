# stdlib
from collections.abc import Callable
import logging
import os
import sys
from typing import Any
from typing import NoReturn
from typing import TextIO

# third party
from loguru import logger
import structlog
from structlog.dev import ConsoleRenderer
from structlog._log_levels import NAME_TO_LEVEL
from structlog.stdlib import BoundLogger
from structlog.types import Processor

timestamper = structlog.processors.TimeStamper(fmt="iso")

processors: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    timestamper,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.CallsiteParameterAdder(
        {
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.LINENO,
            structlog.processors.CallsiteParameter.PROCESS,
            structlog.processors.CallsiteParameter.FILENAME
        }
    ),
    structlog.dev.set_exc_info,
    structlog.processors.format_exc_info,
    structlog.processors.StackInfoRenderer()
]


def _create_kv_formatter( value_style: str, prefix: str = '', postfix: str = '') -> structlog.dev.KeyValueColumnFormatter:
    return structlog.dev.KeyValueColumnFormatter(
        key_style=None,
        value_style=value_style,
        reset_style='\x1b[0m',
        value_repr=str,
        prefix=prefix,
        postfix=postfix
    )


def _fix_console_renderer_columns(renderer: structlog.dev.ConsoleRenderer) -> structlog.dev.ConsoleRenderer:
    gray = '\x1b[1m\x1b[30m'
    blue = '\x1b[1m\x1b[34m'
    cyan = '\x1b[1m\x1b[36m'
    dim_white = '\x1b[2m\x1b[37m'
    white = '\x1b[22m\x1b[37m'
    bright_white = '\x1b[1m\x1b[37m'
    to_left = '\x1b[1D'

    # fix timestamp style
    renderer._columns[0].formatter.value_style = gray

    new_columns = [
        structlog.dev.Column("filename", formatter=_create_kv_formatter(white, prefix=dim_white + '\t::<')),
        structlog.dev.Column("func_name", formatter=_create_kv_formatter(bright_white, prefix=to_left + dim_white + ':')),
        structlog.dev.Column("lineno", formatter=_create_kv_formatter(cyan, prefix=to_left + dim_white + ':', postfix=dim_white + '>')),
        structlog.dev.Column("process", formatter=_create_kv_formatter(blue, prefix='pid[', postfix=']')),
    ]

    renderer._columns.extend(new_columns)

    # move pid to second :)
    renderer._columns = [renderer._columns[0], new_columns[-1]] + renderer._columns[1:-1]

    return renderer


def configure_structlog(log_level: str = "INFO"):
    if isinstance(log_level, str):
        log_level = log_level.lower()
        log_level_no = NAME_TO_LEVEL[log_level]
    elif isinstance(log_level, int):
        log_level_no = log_level
    else:
        logging.getLogger().info(f"Invalid log level: {log_level}, using INFO")
        log_level_no = NAME_TO_LEVEL['info']

    structlog.configure(
        cache_logger_on_first_use=True,
        logger_factory=structlog.stdlib.LoggerFactory(),  # TODO: Support files
        processors=processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(log_level_no),
    )


def configure_std_logging(log_level: str = "INFO", json: bool = False):
    formatter_processors = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]

    if json:
        formatter_renderer = [
            structlog.stdlib.ExtraAdder(),
            structlog.processors.EventRenamer('message'),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        console_renderer = ConsoleRenderer()
        console_renderer = _fix_console_renderer_columns(console_renderer)
        formatter_renderer = [console_renderer]

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=processors,
        processors=formatter_processors + formatter_renderer,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level.upper())

    for _log in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.asgi",
        "fastapi",
    ]:
        logging.getLogger(_log).handlers = [handler]
        logging.getLogger(_log).propagate = False


def get_logger() -> BoundLogger:
    return structlog.stdlib.get_logger('syft')


def start_logger(log_level: str = "info", json: bool = False) -> bool:
    configure_structlog(log_level)
    configure_std_logging(log_level, json)
    return structlog.is_configured()


LOG_FORMAT = "[{time}][{level}][{module}]][{process.id}] {message}"

# logger.remove()
DEFAULT_SINK = None  # "syft_{time}.log"


def remove() -> None:
    logger.remove()


def add(
    sink: None | str | os.PathLike | TextIO | logging.Handler = None,
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
            colorize=False,
            diagnose=True,
            backtrace=True,
            level=level,
        )


def start() -> None:
    add(sink=sys.stderr, level="CRITICAL")
    add(sink=sys.stdout, level="DEBUG")


def stop() -> None:
    logger.stop()


def traceback_and_raise(e: Any, verbose: bool = False) -> NoReturn:
    try:
        if verbose:
            logger.opt(lazy=True).exception(e)
        else:
            logger.opt(lazy=True).critical(e)
    except BaseException as ex:
        logger.debug("failed to print exception", ex)
    if not issubclass(type(e), Exception):
        e = Exception(e)
    raise e


def critical(*args: Any, **kwargs: Any) -> None:
    # caller = inspect.getframeinfo(inspect.stack()[1][0])
    # print(f"critical:{caller.filename}:{caller.function}:{caller.lineno}:{args}")
    return structlog.stdlib.get_logger().critical(*args, **kwargs)


def error(*args: Any, **kwargs: Any) -> None:
    return structlog.stdlib.get_logger().error(*args, **kwargs)


def warning(*args: Any, **kwargs: Any) -> None:
    return structlog.stdlib.get_logger().warning(*args, **kwargs)


def info(*args: Any, **kwargs: Any) -> None:
    return structlog.stdlib.get_logger().info(*args, **kwargs)


def debug(*args: Any) -> None:
    debug_msg = " ".join([str(a) for a in args])
    return structlog.stdlib.get_logger().debug(debug_msg)


def _debug(*args: Any, **kwargs: Any) -> None:
    return structlog.stdlib.get_logger().debug(*args, **kwargs)


def trace(*args: Any, **kwargs: Any) -> None:
    return structlog.stdlib.get_logger().debug(*args, **kwargs)
