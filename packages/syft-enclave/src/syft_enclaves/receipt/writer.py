"""Writing a signed receipt into a finished job's outputs, on the enclave."""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from syft_job.job import JobInfo
from syft_job.models import JobState

from syft_enclaves.receipt.collect import (
    RECEIPT_FILE_NAME,
    build_receipt,
    dataset_entry,
    execution_section,
    job_section,
    results_section,
)
from syft_enclaves.receipt.dsse import sign_receipt

if TYPE_CHECKING:
    from syft_enclaves.client import SyftEnclaveClient

logger = logging.getLogger(__name__)

#: Written into the review dir when the enclave picks the job up to run it.
STARTED_MARKER = "run_started_at"
#: Shipped in place of the receipt when signing one fails, so the submitter
#: sees why instead of a receipt that silently never arrives.
RECEIPT_ERROR_FILE_NAME = "receipt_error.txt"


@dataclass(frozen=True)
class ReceiptSettings:
    """Where the enclave was released from, recorded in every receipt."""

    tinfoil_repo: Optional[str] = None
    tinfoil_release_tag: Optional[str] = None


def mark_started(review_dir: Path) -> None:
    marker = review_dir / STARTED_MARKER
    if not marker.exists():
        marker.write_text(datetime.now(timezone.utc).isoformat())


def write_receipt(
    client: "SyftEnclaveClient", job: JobInfo, settings: ReceiptSettings
) -> Path:
    """Sign a receipt for *job* and save it beside the job's outputs."""
    private_jwks = client._rds.peer_manager.peer_store._ensure_private_keys().to_jwks()
    envelope = sign_receipt(_receipt(client, job, settings), private_jwks)
    path = job.job_review_path / "outputs" / RECEIPT_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2))
    logger.info("Wrote signed receipt for job %s to %s", job.name, path)
    return path


def write_receipt_error(job: JobInfo) -> None:
    """Record the current exception in the job's outputs."""
    path = job.job_review_path / "outputs" / RECEIPT_ERROR_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(traceback.format_exc())


def _receipt(
    client: "SyftEnclaveClient", job: JobInfo, settings: ReceiptSettings
) -> dict[str, Any]:
    return build_receipt(
        job=_job(job),
        data_owners=client.data_owners,
        datasets=_datasets(client, job),
        results=results_section(job.job_review_path / "outputs"),
        execution=_execution(client, job, settings),
    )


def _job(job: JobInfo) -> dict[str, Any]:
    metadata = job.job_metadata
    return job_section(
        name=job.name,
        submitted_by=job.submitted_by,
        submitted_at=job.submitted_at,
        entrypoint=metadata.entrypoint,
        code_dir=job.code_dir,
        code_files=list(metadata.files),
    )


def _datasets(client: "SyftEnclaveClient", job: JobInfo) -> list[dict[str, Any]]:
    entries = []
    for owner, names in sorted((job.job_metadata.datasets or {}).items()):
        for name in sorted(names):
            dataset = client.datasets.get(name, datasite=owner)
            files = [f for f in dataset.private_files if f.name != "syft.pub.yaml"]
            entries.append(dataset_entry(owner, name, files, client.syftbox_folder))
    return entries


def _execution(
    client: "SyftEnclaveClient", job: JobInfo, settings: ReceiptSettings
) -> dict[str, Any]:
    review_dir = job.job_review_path
    public_key = client._rds.peer_manager.peer_store.public_key.identity_key_bytes
    return execution_section(
        started_at=_read_started(review_dir),
        finished_at=JobState.load(review_dir / "state.yaml").completed_at,
        run_public_key=public_key() if callable(public_key) else public_key,
        tinfoil_repo=settings.tinfoil_repo,
        tinfoil_release_tag=settings.tinfoil_release_tag,
    )


def _read_started(review_dir: Path) -> Optional[datetime]:
    marker = review_dir / STARTED_MARKER
    return datetime.fromisoformat(marker.read_text()) if marker.exists() else None
