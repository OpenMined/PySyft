from syft_migration import MigrationRegistry

from ..version import __version__

# Package-local registry for all versioned syft-job objects. The current
# protocol schema is computed from the objects registered into it.
job_registry = MigrationRegistry(
    protocol_name="syft-job",
    package_name="syft-job",
    package_version=__version__,
)
