"""What a receipt says: which code ran, on which data, approved by whom, with
what result, and where.

The trust chain that makes it worth believing is in ``tinfoil/CLAUDE.md``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from syft_enclaves.evidence.tinfoil import (
    TINFOIL_ATTESTATION_PATH,
    TINFOIL_CONFIG_PATH,
    TinfoilProvider,
)

RECEIPT_TYPE = "https://openmined.org/syft-enclave/receipt/v1"
RECEIPT_FILE_NAME = "receipt.dsse.json"


def build_receipt(
    job: dict[str, Any],
    data_owners: list[str],
    datasets: list[dict[str, Any]],
    results: list[dict[str, Any]],
    execution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "_type": RECEIPT_TYPE,
        "job": job,
        "dataOwners": sorted(data_owners),
        "datasets": datasets,
        "results": results,
        "execution": execution,
    }


def file_entry(path: Path, name: str, with_content: bool) -> dict[str, Any]:
    """A file's name and sha256, and its content when *with_content*."""
    data = path.read_bytes()
    entry: dict[str, Any] = {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
    if with_content:
        entry.update(_content(data))
    return entry


def _content(data: bytes) -> dict[str, str]:
    try:
        return {"content": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"content": base64.b64encode(data).decode(), "encoding": "base64"}


def job_section(
    name: str,
    submitted_by: str,
    submitted_at: Optional[str],
    entrypoint: Optional[str],
    code_dir: Path,
    code_files: list[str],
) -> dict[str, Any]:
    """The job, with the full content of every file that was submitted."""
    return {
        "name": name,
        "submittedBy": submitted_by,
        "submittedAt": submitted_at,
        "entrypoint": entrypoint,
        "code": [file_entry(code_dir / f, f, with_content=True) for f in code_files],
    }


def results_section(outputs_dir: Path) -> list[dict[str, Any]]:
    """Every output file except the receipt itself, with its full content."""
    files = sorted(p for p in outputs_dir.rglob("*") if p.is_file())
    return [
        file_entry(p, str(p.relative_to(outputs_dir)), with_content=True)
        for p in files
        if p.name != RECEIPT_FILE_NAME
    ]


def dataset_entry(
    owner: str, name: str, files: list[Path], root: Path
) -> dict[str, Any]:
    """A dataset's private files, as paths under *root* and their hashes."""
    root = root.resolve()
    return {
        "owner": owner,
        "name": name,
        "files": [
            file_entry(f, str(f.resolve().relative_to(root)), with_content=False)
            for f in sorted(files)
        ],
    }


def execution_section(
    started_at: Optional[datetime],
    finished_at: Optional[datetime],
    run_public_key: bytes,
    tinfoil_repo: Optional[str],
    tinfoil_release_tag: Optional[str],
) -> dict[str, Any]:
    """Where and when it ran, and the key that signs the receipt."""
    platform = _platform_section(tinfoil_repo, tinfoil_release_tag)
    return {
        **platform,
        "runId": uuid.uuid4().hex,
        "startedAt": _iso(started_at),
        "finishedAt": _iso(finished_at),
        "runPublicKey": run_public_key.hex(),
    }


def _platform_section(repo: Optional[str], tag: Optional[str]) -> dict[str, Any]:
    if not TinfoilProvider.detect():
        return {"platform": "local", "attestation": None}
    config = TINFOIL_CONFIG_PATH.read_bytes()
    return {
        "platform": "tinfoil-containers",
        "cvmVersion": (yaml.safe_load(config) or {}).get("cvm-version"),
        "configDigest": hashlib.sha256(config).hexdigest(),
        "attestation": _attestation(repo, tag),
    }


def _attestation(repo: Optional[str], tag: Optional[str]) -> dict[str, Any]:
    """The hardware report as the enclave booted with it, and its reference.

    The report does not name the run key: on Tinfoil its report data is the
    shim's TLS key. The run key is tied to it by the claims the enclave signs
    over a connection pinned to that TLS key (``attestation/https.py``).
    """
    document = json.loads(TINFOIL_ATTESTATION_PATH.read_text())
    kind = str(document.get("format", ""))
    return {
        "type": "tdx" if "tdx" in kind else "sev-snp",
        "format": document.get("format"),
        "quote": document.get("body"),
        "referenceValue": {
            "source": "sigstore",
            "repo": f"github.com/{repo}" if repo else None,
            "tag": tag,
        },
    }


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat().replace("+00:00", "Z") if value else None
