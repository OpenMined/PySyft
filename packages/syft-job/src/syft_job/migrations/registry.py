from syft_migration import MigrationRegistry

from ..version import PACKAGE_NAME, __version__

# Hardcoded, language-agnostic identifier for the syft-job protocol;
# intentionally distinct from the package name.
PROTOCOL_NAME = "syft-job"

# Incrementing version of the job protocol. Protocol 0 is the last release
# without versioning (<= 0.1.38, no v<n> path segment); protocol >= 1 stores
# jobs under a v<n> segment after the peer email (see config.protocol_dir_name).
# Protocol 2 adds api calls: requests/ and RequestMetadata.
JOB_PROTOCOL_VERSION = "2"

# The first job protocol with api calls. A DS can call an api of a DO only
# when the two negotiate at least this protocol.
FIRST_PROTOCOL_WITH_REQUESTS = "2"

# Oldest job protocol this release still reads. "0" refuses no peer. Raise it
# only when the code drops support for a released protocol, because a peer below
# the floor cannot exchange jobs with this release.
MIN_SUPPORTED_JOB_PROTOCOL_VERSION = "0"

# Package-local registry for all versioned syft-job objects. The current
# protocol schema is computed from the objects registered into it.
job_registry = MigrationRegistry(
    protocol_name=PROTOCOL_NAME,
    package_name=PACKAGE_NAME,
    package_version=__version__,
    protocol_version=JOB_PROTOCOL_VERSION,
    min_supported_protocol_version=MIN_SUPPORTED_JOB_PROTOCOL_VERSION,
)
