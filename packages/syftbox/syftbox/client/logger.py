import sys
from datetime import datetime
from pathlib import Path
from shutil import make_archive
from typing import Union

import loguru
from loguru import logger

from syftbox.lib.constants import DEFAULT_LOGS_DIR
from syftbox.lib.types import PathLike, to_path

LOGS_FORMAT = loguru


def setup_logger(
    level: Union[str, int] = "DEBUG",
    log_dir: PathLike = DEFAULT_LOGS_DIR,
    keep_logs: int = 10,
) -> None:
    logger.remove()
    logger.add(level=level, sink=sys.stderr, diagnose=False, backtrace=False)

    # new file per run - no rotation needed
    # always log debug level
    log_file = Path(log_dir, f"syftbox_{int(datetime.now().timestamp())}.log")
    logger.add(
        log_file,
        level="DEBUG",
        rotation=None,
        compression=None,
        colorize=True,
    )

    # keep last 5 logs
    logs_to_delete = sorted(Path(log_dir).glob("syftbox_*.log"))[:-keep_logs]
    for log in logs_to_delete:
        try:
            log.unlink()
        except Exception:
            pass


def zip_logs(output_path: PathLike, log_dir: PathLike = DEFAULT_LOGS_DIR) -> str:
    logs_folder = to_path(log_dir)
    return make_archive(str(output_path), "zip", logs_folder)
