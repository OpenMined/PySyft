"""Export the release artifacts for the current syft-job version.

Run on EVERY release (uv run python -m syft_job.migrations.export_release_artifact):
always writes the package release info; additionally writes the protocol
artifact when this release introduces a new protocol version.
"""

import sys

from ..version import __version__
from .history import PACKAGE_ARTIFACTS_DIR, PROTOCOLS_DIR
from .registry import JOB_PROTOCOL_VERSION, job_registry


def main() -> None:
    # Import the models so every versioned object is registered.
    import syft_job  # noqa: F401

    if job_registry.protocol_changed_without_bump():
        sys.exit(
            "The job protocol changed compared to the released "
            f"protocol-{JOB_PROTOCOL_VERSION}.json; bump JOB_PROTOCOL_VERSION "
            "in syft_job/migrations/registry.py before releasing."
        )

    info_path = PACKAGE_ARTIFACTS_DIR / f"syft-job-{__version__}.json"
    job_registry.compute_released_package_protocol_info().save(info_path)
    print(f"Wrote {info_path}")

    protocol_path = PROTOCOLS_DIR / f"protocol-{JOB_PROTOCOL_VERSION}.json"
    if not protocol_path.exists():
        job_registry.compute_released_protocol().save(protocol_path)
        print(f"Wrote {protocol_path} (new protocol version)")


if __name__ == "__main__":
    main()
