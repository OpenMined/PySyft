"""What a receipt says: which code ran, on which data, approved by whom, with
what result, and where.

The receipt is an in-toto statement. The enclave writes every section of it
except the ones the job claims for itself (``receipt/claims.py``).

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
from syft_enclaves.receipt.claims import CLAIMS_FILE_NAME
from syft_enclaves.receipt.dsse import canonical_json

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://openmined.org/syft-enclave/receipt/v2"
RECEIPT_FILE_NAME = "receipt.dsse.json"
#: Everything a party can be sent: the job's output files, and this receipt.
ALL_GRANTS = ["results", "receipt"]


def build_receipt(claims: dict[str, Any], **sections: Any) -> dict[str, Any]:
    """The statement: the job's *claims*, then the enclave's own *sections*.

    The enclave's sections go last, so a claim can never replace one of them.
    """
    predicate = {k: claims[k] for k in ("model", "eval", "results") if k in claims}
    predicate.update(sections)
    return {
        "_type": STATEMENT_TYPE,
        "subject": claims.get("subject", []),
        "predicateType": PREDICATE_TYPE,
        "predicate": predicate,
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


def outputs_section(outputs_dir: Path) -> list[dict[str, Any]]:
    """Every output file with its full content, except the receipt's own files."""
    files = sorted(p for p in outputs_dir.rglob("*") if p.is_file())
    return [
        file_entry(p, str(p.relative_to(outputs_dir)), with_content=True)
        for p in files
        if p.name not in (RECEIPT_FILE_NAME, CLAIMS_FILE_NAME)
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


def parties_section(
    submitted_by: str, datasets: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Who took part: the submitter, and each data owner with their datasets."""
    owners = [
        {"role": "data_owner", "email": owner, "datasets": sorted(names)}
        for owner, names in sorted(datasets.items())
    ]
    return [{"role": "submitter", "email": submitted_by}, *owners]


def consent_section(
    code: list[dict[str, Any]], approvals: dict[str, Optional[datetime]]
) -> dict[str, Any]:
    """Who approved the code, and when. The digest names the code they approved."""
    return {
        "manifestDigest": hashlib.sha256(canonical_json(code)).hexdigest(),
        "approvals": [
            {"party": party, "approvedAt": _iso(approved_at)}
            for party, approved_at in sorted(approvals.items())
        ],
    }


def policy_section(
    submitted_by: str, data_owners: list[str], share_with_owners: bool
) -> dict[str, Any]:
    """Who is sent what: the submitter everything, the owners only if shared."""
    owner_grants = ALL_GRANTS if share_with_owners else []
    policy = [
        {"party": owner, "grants": owner_grants}
        for owner in sorted(data_owners)
        if owner != submitted_by
    ]
    return {"outputPolicy": [{"party": submitted_by, "grants": ALL_GRANTS}, *policy]}


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
    parsed = yaml.safe_load(config) or {}
    return {
        "platform": "tinfoil-containers",
        "cvmVersion": parsed.get("cvm-version"),
        "configDigest": hashlib.sha256(config).hexdigest(),
        "runtimeImage": _image_digest(parsed),
        "attestation": _attestation(repo, tag),
    }


def _image_digest(config: dict[str, Any]) -> Optional[dict[str, str]]:
    """The digest the config pins the enclave's container image to."""
    containers = config.get("containers") or [{}]
    image = str(containers[0].get("image", ""))
    if "@" not in image:
        return None
    return {"scheme": "oci/1", "digest": image.split("@", 1)[1]}


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
