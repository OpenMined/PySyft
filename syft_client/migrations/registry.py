from syft_migration import MigrationRegistry, MigrationService

from syft_client.version import SYFT_CLIENT_VERSION

PACKAGE_NAME = "syft-client"

# Hardcoded, language-agnostic identifier for the syft-client protocol;
# intentionally distinct from the package name.
PROTOCOL_NAME = "syft-client"

# Incrementing version of the syft-client protocol. Protocol 0 is the last
# release without per-object versioning (<= 0.1.117, files carry no
# canonical_name/version identity fields); protocol >= 1 serializes identity
# fields on every versioned object.
SYFT_CLIENT_PROTOCOL_VERSION = "1"

# Oldest syft-client protocol this release still reads. "0" refuses no peer.
# Raise it only when the code drops support for a released protocol, because a
# peer below the floor cannot exchange syft-client messages with this release.
MIN_SUPPORTED_SYFT_CLIENT_PROTOCOL_VERSION = "0"

# Package-local registry for all versioned syft-client objects. The current
# protocol schema is computed from the objects registered into it.
client_registry = MigrationRegistry(
    protocol_name=PROTOCOL_NAME,
    package_name=PACKAGE_NAME,
    package_version=SYFT_CLIENT_VERSION,
    protocol_version=SYFT_CLIENT_PROTOCOL_VERSION,
    min_supported_protocol_version=MIN_SUPPORTED_SYFT_CLIENT_PROTOCOL_VERSION,
)

# Shared service for loading/migrating syft-client objects.
client_migration_service = MigrationService(registry=client_registry)


def load_as_latest(data: dict, canonical_name: str) -> object:
    """Load ``data`` (defaulting identity fields for protocol-0 files, which
    predate them and are all version 1) and migrate to the latest version."""
    data.setdefault("canonical_name", canonical_name)
    data.setdefault("version", "1")
    obj = client_migration_service.load(data)
    return client_migration_service.migrate(
        obj, client_registry.latest_version(canonical_name)
    )
