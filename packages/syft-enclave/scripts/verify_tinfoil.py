"""Verify a live Tinfoil enclave the way a data owner would.

Fetches the enclave's attestation document from the shim and appraises it
against the signed release of a config repo. Used by ``just tinfoil-verify``;
a data owner in a notebook calls ``client.attest_peer(...)`` instead, which
reads the same evidence from the peer's ``SYFT_version.json`` on Drive.

Needs the optional extra: ``uv pip install "syft-enclave[tinfoil]"``.
"""

from __future__ import annotations

import argparse
import sys

from syft_enclaves.attestation import AttestationError
from syft_enclaves.attestation.envelope import tinfoil_evidence
from syft_enclaves.attestation.tinfoil import (
    DEFAULT_TINFOIL_CONFIG_REPO,
    TinfoilAppraisalPolicy,
    verify_tinfoil_evidence,
)
from syft_enclaves.optional_deps import require

ATTESTATION_PATH = "/.well-known/tinfoil-attestation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "host", help="enclave hostname, e.g. x.y.containers.tinfoil.dev"
    )
    parser.add_argument("--repo", default=DEFAULT_TINFOIL_CONFIG_REPO)
    parser.add_argument(
        "--tag",
        default=None,
        help="pin a release tag; omit to appraise against the latest release",
    )
    parser.add_argument(
        "--expected-image-digest",
        default=None,
        help="'sha256:...' digest to pin; omit to skip the image check",
    )
    parser.add_argument(
        "--expected-enclave-email",
        default=None,
        help="the datasite the enclave should be running as",
    )
    parser.add_argument(
        "--expected-data-owners",
        default=None,
        type=lambda v: [e.strip() for e in v.split(",") if e.strip()],
        help="comma-separated emails whose approval must gate a job",
    )
    parser.add_argument("--container-name", default="syft-enclave")
    return parser.parse_args()


def fetch_document(host: str) -> dict:
    requests = require(
        "requests",
        extra="tinfoil",
        feature="Verifying a Tinfoil enclave",
        docs="packages/syft-enclave/docs/tinfoil_deployment.md",
    )
    response = requests.get(f"https://{host}{ATTESTATION_PATH}", timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parse_args()
    policy = TinfoilAppraisalPolicy(
        repo=args.repo,
        release_tag=args.tag,
        expected_image_digest=args.expected_image_digest,
        expected_data_owners=args.expected_data_owners,
        expected_email=args.expected_enclave_email,
        container_name=args.container_name,
        # This script checks a live host, often before the digest and the
        # data-owner list are known, so it opts out rather than refusing to
        # build a policy.
        allow_unpinned=not (
            args.expected_image_digest
            and args.expected_data_owners
            and args.expected_enclave_email
        ),
    )
    try:
        verify_tinfoil_evidence(tinfoil_evidence(fetch_document(args.host)), policy)
    except AttestationError:
        # verify_tinfoil_evidence already printed the full checklist.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
