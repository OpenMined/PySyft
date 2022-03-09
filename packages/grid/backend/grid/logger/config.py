"""This file defines the configuration for `loguru` which is used as the python logging client.
For more information refer to `loguru` documentation: https://loguru.readthedocs.io/en/stable/overview.html
"""

# stdlib
from datetime import time
from datetime import timedelta
from enum import Enum
from functools import lru_cache
from typing import Optional
from typing import Union

# third party
from pydantic import BaseSettings


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

    LOGURU_LEVEL: str = LogLevel.DEBUG.value
    LOGURU_SINK: Optional[str] = "/var/log/pygrid/grid.log"
    LOGURU_COMPRESSION: Optional[str]
    LOGURU_ROTATION: Union[
        Optional[str], Optional[int], Optional[time], Optional[timedelta]
    ]
    LOGURU_RETENTION: Union[Optional[str], Optional[int], Optional[timedelta]]
    LOGURU_COLORIZE: Optional[bool] = True
    LOGURU_SERIALIZE: Optional[bool] = False
    LOGURU_BACKTRACE: Optional[bool] = True
    LOGURU_DIAGNOSE: Optional[bool] = False
    LOGURU_ENQUEUE: Optional[bool] = True
    LOGURU_AUTOINIT: Optional[bool] = False


@lru_cache()
def get_log_config() -> LogConfig:
    """Returns the configuration for the logging client.

    Returns:
        LogConfig: configuration for the logging client.
    """
    return LogConfig()
