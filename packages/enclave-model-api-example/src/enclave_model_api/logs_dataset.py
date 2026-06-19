"""Inference-logs dataset on the enclave's own datasite.

Only the synthetic mock sample (schema reference for researchers) goes to
Drive. The private dir is the live log sink written by the inference server;
``private/`` is excluded from sync and nothing ever shares it — data owners
jointly approve computations on the logs but can never download them.
"""

import json
import tempfile
from pathlib import Path

from enclave_model_api.log_writer import LOG_FILE_NAME

LOGS_DATASET_SUMMARY = (
    "Inference request logs (prompt, completion, stats) collected by the "
    "enclave inference service. Private data never leaves the enclave; "
    "analysis jobs require approval from all data owners."
)

SAMPLE_RECORD = {
    "id": "00000000-0000-0000-0000-000000000000",
    "timestamp": "1970-01-01T00:00:00+00:00",
    "prompt": "<synthetic sample - not a real request>",
    "completion": "<synthetic sample completion>",
    "stats": {"elapsed": 0.0},
}


def _seed_dirs() -> tuple[Path, Path]:
    """Create temp mock (one synthetic record) and private (empty log) dirs."""
    mock_dir = Path(tempfile.mkdtemp(prefix="inference_logs_mock_"))
    (mock_dir / f"sample_{LOG_FILE_NAME}").write_text(json.dumps(SAMPLE_RECORD) + "\n")
    private_dir = Path(tempfile.mkdtemp(prefix="inference_logs_private_"))
    (private_dir / LOG_FILE_NAME).write_text("")
    return mock_dir, private_dir


def ensure_logs_dataset(client, name: str) -> Path:
    """Create the logs dataset if missing; return its private dir (the log sink).

    Idempotent so reboots with fresh_state=false keep the existing logs.
    """
    config = client.datasets.syftbox_config
    if (config.get_my_mock_dataset_dir(name) / "dataset.yaml").exists():
        return config.private_dir_for_my_dataset(name)

    mock_dir, private_dir = _seed_dirs()
    client.create_dataset(
        name=name,
        mock_path=mock_dir,
        private_path=private_dir,
        summary=LOGS_DATASET_SUMMARY,
        users="any",
        upload_private=False,
        sync=False,
    )
    client.sync()
    return config.private_dir_for_my_dataset(name)
