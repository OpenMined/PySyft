from syft_migration import MigrationRegistry

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

# Package-local registry for all versioned syft-client objects. The current
# protocol schema is computed from the objects registered into it.
client_registry = MigrationRegistry(
    protocol_name=PROTOCOL_NAME,
    package_name=PACKAGE_NAME,
    package_version=SYFT_CLIENT_VERSION,
    protocol_version=SYFT_CLIENT_PROTOCOL_VERSION,
)
