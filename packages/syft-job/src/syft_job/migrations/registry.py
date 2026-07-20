from syft_migration import MigrationRegistry

from ..version import PACKAGE_NAME, __version__

# Hardcoded, language-agnostic identifier for the syft-job protocol;
# intentionally distinct from the package name.
PROTOCOL_NAME = "syft-job"

# Incrementing version of the job protocol. Protocol 0 is the last release
# without versioning (<= 0.1.38, no v<n> path segment); protocol >= 1 stores
# jobs under a v<n> segment after the peer email (see config.protocol_dir_name).
JOB_PROTOCOL_VERSION = "1"

# Package-local registry for all versioned syft-job objects. The current
# protocol schema is computed from the objects registered into it.
job_registry = MigrationRegistry(
    protocol_name=PROTOCOL_NAME,
    package_name=PACKAGE_NAME,
    package_version=__version__,
    protocol_version=JOB_PROTOCOL_VERSION,
)
