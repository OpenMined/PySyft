import json
import logging
import sys
from pathlib import Path
from typing import Union

from loguru import logger

custom_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{message}</cyan>"

ANALYTICS_EVENT = "analytics_event"


def _default_logger_filter(record: dict) -> bool:
    return record["extra"].get("event_type") != ANALYTICS_EVENT


def _analytics_logger_filter(record: dict) -> bool:
    return record["extra"].get("event_type") == ANALYTICS_EVENT


analytics_logger = logger.bind(event_type=ANALYTICS_EVENT)


def analytics_formatter(record: dict) -> str:
    serialized = json.dumps(record["extra"])
    record["extra"]["serialized"] = serialized
    return "{extra[serialized]}\n"


def setup_logger(logs_folder: Path, level: Union[str, int] = "DEBUG") -> None:
    logs_folder.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # Standard server logs
    logger.add(
        level=level,
        sink=sys.stderr,
        diagnose=False,
        backtrace=False,
        format=custom_format,
        filter=_default_logger_filter,
    )

    logger.add(
        logs_folder / "server.log",
        rotation="100 MB",  # Rotate after the log file reaches 100 MB
        retention=2,  # Keep only the last 1 log files
        compression="zip",  # Usually, 10x reduction in file size
        filter=_default_logger_filter,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",  # matches the log format printed in the console
    )

    # Dedicated logger for analytics events
    # example usage: user_event_logger.info("User logged in")
    logger.add(
        logs_folder / "analytics.log",
        rotation="100 MB",
        compression="zip",
        format=analytics_formatter,
        filter=_analytics_logger_filter,
    )

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True

    logger.info(f"Logger set up. Saving logs to {logs_folder}")
