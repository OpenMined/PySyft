"""Export the release artifacts for the current syft-dataset version.

Run on EVERY release (uv run python scripts/export_release_artifact.py):
always writes the package release info; additionally writes the protocol
artifact when this release introduces a new protocol version.

Artifacts are frozen once written. Running this again for the same version
writes nothing and succeeds, so a release can re-run it safely.
"""

import sys

from syft_datasets.migrations.history import PACKAGE_ARTIFACTS_DIR, PROTOCOLS_DIR
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION, dataset_registry
from syft_datasets.version import __version__


def main() -> None:
    # Import the models so every versioned object is registered.
    import syft_datasets  # noqa: F401

    if dataset_registry.protocol_bump_missing():
        sys.exit(
            "The dataset protocol changed since the released "
            f"protocol-{dataset_registry.latest_released_protocol_version()}.json; "
            "bump DATASET_PROTOCOL_VERSION in "
            "syft_datasets/migrations/registry.py before releasing."
        )

    if dataset_registry.protocol_changed_without_bump():
        sys.exit(
            "The dataset protocol changed compared to the released "
            f"protocol-{DATASET_PROTOCOL_VERSION}.json; bump "
            "DATASET_PROTOCOL_VERSION in syft_datasets/migrations/registry.py "
            "before releasing."
        )

    PACKAGE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)

    info_path = PACKAGE_ARTIFACTS_DIR / f"syft-dataset-{__version__}.json"
    protocol_path = PROTOCOLS_DIR / f"protocol-{DATASET_PROTOCOL_VERSION}.json"

    if info_path.exists():
        print(f"Package artifact already present: {info_path}")
    else:
        dataset_registry.compute_released_package_protocol_info().save(info_path)
        print(f"Wrote {info_path}")

    if protocol_path.exists():
        print(f"Protocol artifact already present: {protocol_path}")
    else:
        dataset_registry.compute_released_protocol().save(protocol_path)
        print(f"Wrote {protocol_path} (new protocol version)")


if __name__ == "__main__":
    main()
