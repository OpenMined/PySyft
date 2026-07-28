"""Export the release artifacts for the current syft-client version.

Run on EVERY release (uv run python scripts/export_release_artifact.py):
always writes the package release info; additionally writes the protocol
artifact when this release introduces a new protocol version.
"""

import sys

from syft_client.migrations.history import PACKAGE_ARTIFACTS_DIR, PROTOCOLS_DIR
from syft_client.migrations.registry import (
    SYFT_CLIENT_PROTOCOL_VERSION,
    client_registry,
)
from syft_client.version import SYFT_CLIENT_VERSION


def main() -> None:
    # Import the package so every versioned object is registered.
    import syft_client  # noqa: F401

    if client_registry.protocol_changed_without_bump():
        sys.exit(
            "The syft-client protocol changed compared to the released "
            f"protocol-{SYFT_CLIENT_PROTOCOL_VERSION}.json; bump "
            "SYFT_CLIENT_PROTOCOL_VERSION in syft_client/migrations/registry.py "
            "before releasing."
        )

    info_path = PACKAGE_ARTIFACTS_DIR / f"syft-client-{SYFT_CLIENT_VERSION}.json"
    protocol_path = PROTOCOLS_DIR / f"protocol-{SYFT_CLIENT_PROTOCOL_VERSION}.json"
    need_info = not info_path.exists()
    need_protocol = not protocol_path.exists()

    # Single exit when there is nothing left to write
    if not need_info and not need_protocol:
        sys.exit(
            f"Release artifacts already present:\n"
            f"  {info_path}\n"
            f"  {protocol_path}\n"
            "They are frozen once written. Bump SYFT_CLIENT_VERSION (and "
            "SYFT_CLIENT_PROTOCOL_VERSION if the protocol changed) before "
            "exporting again."
        )

    if need_info:
        client_registry.compute_released_package_protocol_info().save(info_path)
        print(f"Wrote {info_path}")
    else:
        print(f"Package artifact already present: {info_path}")

    if need_protocol:
        client_registry.compute_released_protocol().save(protocol_path)
        print(f"Wrote {protocol_path} (new protocol version)")


if __name__ == "__main__":
    main()
