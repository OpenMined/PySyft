import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from syftbox.server.db.file_store import FileStore
from syftbox.server.logger import analytics_logger


def to_jsonable_dict(obj: dict) -> dict:
    """
    Convert log record to a JSON serializable dictionary.
    """
    result: dict = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            result[key] = to_jsonable_dict(value)
        elif isinstance(value, BaseModel):
            result[key] = value.model_dump(mode="json")
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, Path):
            result[key] = value.as_posix()
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        else:
            result[key] = str(value)

    return result


def log_analytics_event(
    endpoint: str,
    email: Optional[str],
    message: str = "",
    **kwargs: Any,
) -> None:
    """
    Log an event to the analytics logger.
    """
    email = email or "anonymous"

    try:
        extra = {
            "email": email,
            "endpoint": endpoint,
            "timestamp": datetime.now(timezone.utc),
            **kwargs,
        }
        extra = to_jsonable_dict(extra)
        analytics_logger.bind(**extra).info(message)
    except Exception as e:
        logger.error(f"Failed to log event: {e}")


def log_file_change_event(
    endpoint: str,
    email: str,
    relative_path: Path,
    file_store: FileStore,
) -> None:
    """
    Log a file change event to the analytics logger.
    """
    try:
        metadata = file_store.get_metadata(relative_path, email, skip_permission_check=True)
        log_analytics_event(
            endpoint=endpoint,
            email=email,
            file_metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Failed to log file change event: {e}")


def _parse_analytics_file(file_path: Path) -> list[dict]:
    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path, "r") as zfile:
            with zfile.open(zfile.namelist()[0]) as f:
                content = f.read().decode("utf-8")
    else:
        with open(file_path, "r") as f:
            content = f.read()

    events = []
    for line in content.split("\n"):
        if not line:
            continue

        try:
            event = json.loads(line)
            event["timestamp"] = datetime.fromisoformat(event["timestamp"])
            events.append(event)
        except Exception as e:
            logger.error(f"Failed to parse event: {e}")

    return events


def parse_analytics_logs(logs_dir: Path) -> list[dict]:
    # Load current log and all archived logs
    log_files = list(logs_dir.glob("analytics.log")) + list(logs_dir.glob("analytics*.zip"))
    logger.info(f"Loading logs from: {[f.as_posix() for f in log_files]}")
    events = []
    for log_file in log_files:
        events.extend(_parse_analytics_file(log_file))

    events = sorted(events, key=lambda x: x["timestamp"])
    return events
