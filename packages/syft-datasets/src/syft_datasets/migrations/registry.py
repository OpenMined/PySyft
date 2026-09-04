from syft_migration import MigrationRegistry

from ..version import PACKAGE_NAME, __version__

# Hardcoded, language-agnostic identifier for the syft-dataset protocol;
# intentionally distinct from the package name.
PROTOCOL_NAME = "syft-dataset"

# Incrementing version of the dataset protocol. Protocol 0 is the last release
# without versioning (syft-client 0.1.117 / syft-dataset 0.1.20, no v<n> path
# segment); protocol >= 1 stores datasets under a v<n> segment after the
# syft_datasets folder (see config.protocol_dir_name).
DATASET_PROTOCOL_VERSION = "1"

# Oldest dataset protocol this release still reads. "0" refuses no peer. Raise it
# only when the code drops support for a released protocol, because a peer below
# the floor cannot exchange datasets with this release.
MIN_SUPPORTED_DATASET_PROTOCOL_VERSION = "0"

# Package-local registry for all versioned syft-dataset objects. The current
# protocol schema is computed from the objects registered into it.
dataset_registry = MigrationRegistry(
    protocol_name=PROTOCOL_NAME,
    package_name=PACKAGE_NAME,
    package_version=__version__,
    protocol_version=DATASET_PROTOCOL_VERSION,
    min_supported_protocol_version=MIN_SUPPORTED_DATASET_PROTOCOL_VERSION,
)
