"""Append-only JSONL log of inference requests.

Written by the inference server directly into the private dir of the logs
dataset on the enclave's own datasite. ``private/`` paths are excluded from
sync, so these records never reach Drive; researchers analyse them through
jobs that both data owners must approve.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE_NAME = "requests.jsonl"


def build_log_record(prompt: str, completion: str, stats: dict) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "completion": completion,
        "stats": stats,
    }


def append_log_record(logs_dir: Path | str, record: dict) -> Path:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / LOG_FILE_NAME
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    return log_file
