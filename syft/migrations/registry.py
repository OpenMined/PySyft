from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    MigrationService,
)

from syft.version import SYFT_VERSION

PACKAGE_NAME = "syft"

# Hardcoded, language-agnostic identifier for the syft protocol. A peer reads it
# as a key in SYFT_version.json, so it changes only with the released artifacts
# that carry it (history/).
PROTOCOL_NAME = "syft"

# Incrementing version of the syft protocol.
SYFT_CLIENT_PROTOCOL_VERSION = "0"

# Oldest syft protocol this release still reads. "0" refuses no peer.
# Raise it only when the code drops support for a released protocol, because a
# peer below the floor cannot exchange syft messages with this release.
MIN_SUPPORTED_SYFT_CLIENT_PROTOCOL_VERSION = "0"

# Package-local registry for all versioned syft objects. The current
# protocol schema is computed from the objects registered into it.
client_registry = MigrationRegistry(
    protocol_name=PROTOCOL_NAME,
    package_name=PACKAGE_NAME,
    package_version=SYFT_VERSION,
    protocol_version=SYFT_CLIENT_PROTOCOL_VERSION,
    min_supported_protocol_version=MIN_SUPPORTED_SYFT_CLIENT_PROTOCOL_VERSION,
)

# Shared service for loading/migrating syft objects.
client_migration_service = MigrationService(registry=client_registry)


def load_current(data: dict, canonical_name: str) -> MigratableObject:
    """Load ``data`` as a ``canonical_name`` and migrate it to the latest version.

    Raises:
        MigrationError: when ``data`` is not a ``canonical_name`` of a version this
            release knows, such as a file written by an unsupported syft release.
    """
    found = data.get("canonical_name")
    if found != canonical_name:
        raise MigrationError(
            f"Expected a {canonical_name}, found "
            f"{found if found else 'data without identity fields'}"
        )
    obj = client_migration_service.load(data)
    return client_migration_service.migrate(
        obj, client_registry.latest_version(canonical_name)
    )
