from syft_migration import MigrationRegistry

from ..version import PACKAGE_NAME, __version__

# Hardcoded, language-agnostic identifier for the syft-job protocol;
# intentionally distinct from the package name.
PROTOCOL_NAME = "syft-job"

# Package-local registry for all versioned syft-job objects. The current
# protocol schema is computed from the objects registered into it.
job_registry = MigrationRegistry(
    protocol_name=PROTOCOL_NAME,
    package_name=PACKAGE_NAME,
    package_version=__version__,
)
