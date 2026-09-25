"""The parts of a receipt only the job knows: what it evaluated, and how.

A job writes them to ``outputs/receipt_claims.json``. They are only as true as
the job's code, which every data owner read and approved, and whose full
content the receipt carries next to them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLAIMS_FILE_NAME = "receipt_claims.json"
#: The only top-level keys a job may write. Everything else in the receipt is
#: written by the enclave, and a job must not be able to overwrite it.
JOB_CLAIM_KEYS = ("subject", "model", "eval", "results")


class ReceiptClaimsError(Exception):
    """The job's claims file is not something the receipt can include."""


def read_job_claims(outputs_dir: Path) -> dict[str, Any]:
    """The job's claims, or nothing when the job wrote none."""
    path = outputs_dir / CLAIMS_FILE_NAME
    if not path.exists():
        return {}
    try:
        claims = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ReceiptClaimsError(f"{CLAIMS_FILE_NAME} is not valid JSON: {e}") from e
    if not isinstance(claims, dict):
        raise ReceiptClaimsError(f"{CLAIMS_FILE_NAME} must hold a JSON object")
    unknown = sorted(set(claims) - set(JOB_CLAIM_KEYS))
    if unknown:
        raise ReceiptClaimsError(
            f"{CLAIMS_FILE_NAME} may only set {list(JOB_CLAIM_KEYS)}, not {unknown}"
        )
    return claims
