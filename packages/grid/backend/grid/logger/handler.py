# future
from __future__ import annotations

# stdlib
from functools import lru_cache
import logging
from pprint import pformat
import sys

# third party
import loguru
from loguru import logger

# relative
from .config import get_log_config


class LogHandler:
    def __init__(self) -> None:
        self.config = get_log_config()

    def format_record(self, record: loguru.Record) -> str:
        """
        Custom loguru log message format for handling JSON (in record['extra'])
        """
        format_string: str = self.config.LOGURU_FORMAT

        if record["extra"] is not None:
            for key in record["extra"].keys():
                record["extra"][key] = pformat(
                    record["extra"][key], indent=2, compact=False, width=88
                )
                format_string += "\n{extra[" + key + "]}"

        format_string += "<level>{exception}</level>\n"

        return format_string

    def init_logger(self) -> None:
        """
        Redirects all registered std logging handlers to a loguru sink.
        Call init_logger() on fastapi startup.
        """
        intercept_handler = InterceptHandler()

        # Generalizes log level for all root loggers, including third party
        logging.root.setLevel(self.config.LOGURU_LEVEL)
        logging.root.handlers = [intercept_handler]

        for log in logging.root.manager.loggerDict.keys():
            log_instance = logging.getLogger(log)
            log_instance.handlers = []
            log_instance.propagate = True

        logger.configure(
            handlers=[
                {
                    "sink": sys.stdout,
                    "level": self.config.LOGURU_LEVEL,
                    "serialize": self.config.LOGURU_SERIALIZE,
                    "format": self.format_record,
                }
            ]
        )

        try:
            if (
                self.config.LOGURU_SINK is not ("sys.stdout" or "sys.stderr")
                and self.config.LOGURU_SINK is not None
            ):
                logger.add(
                    self.config.LOGURU_SINK,
                    retention=self.config.LOGURU_RETENTION,
                    rotation=self.config.LOGURU_ROTATION,
                    compression=self.config.LOGURU_COMPRESSION,
                )
                logger.debug(f"Logging to {self.config.LOGURU_SINK}")

        except Exception as err:
            logger.debug(
                f"Failed creating a new sink. Check your log config. error: {err}"
            )


class InterceptHandler(logging.Handler):
    """
    Check https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # type: ignore
            frame = frame.f_back  # type: ignore
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


@lru_cache()
def get_log_handler() -> LogHandler:
    return LogHandler()
