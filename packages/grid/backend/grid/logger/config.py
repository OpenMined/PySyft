"""This file defines the configuration for `loguru` which is used as the python logging client.
For more information refer to `loguru` documentation: https://loguru.readthedocs.io/en/stable/overview.html
"""

# stdlib
from datetime import time
from datetime import timedelta
from enum import Enum
from functools import lru_cache

# third party
from pydantic_settings import BaseSettings


# LOGURU_LEVEL type for version>3.8
class LogLevel(Enum):
    """Types of logging levels."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogConfig(BaseSettings):
    """Configuration for the logging client."""

    # Logging format
    LOGURU_FORMAT: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>: "
        "<level>{message}</level>"
    )

    LOGURU_LEVEL: str = LogLevel.INFO.value
    LOGURU_SINK: str | None = "/var/log/pygrid/grid.log"
    LOGURU_COMPRESSION: str | None = None
    LOGURU_ROTATION: str | int | time | timedelta | None = None
    LOGURU_RETENTION: str | int | timedelta | None = None
    LOGURU_COLORIZE: bool | None = True
    LOGURU_SERIALIZE: bool | None = False
    LOGURU_BACKTRACE: bool | None = True
    LOGURU_DIAGNOSE: bool | None = False
    LOGURU_ENQUEUE: bool | None = True
    LOGURU_AUTOINIT: bool | None = False


@lru_cache
def get_log_config() -> LogConfig:
    """Returns the configuration for the logging client.

    Returns:
        LogConfig: configuration for the logging client.
    """
    return LogConfig()
