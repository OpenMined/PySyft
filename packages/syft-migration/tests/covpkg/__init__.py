"""A throwaway package the coverage checks run against.

Holds one object that is registered correctly, one declared in this ``__init__``
module (which the module-prefix scan must still see), and one registered into a
second registry, standing in for an object filed under the wrong package.
"""

from syft_migration import MigratableObject, MigrationRegistry

covered_registry = MigrationRegistry(
    protocol_name="covpkg-proto",
    package_name="covpkg",
    package_version="1.0.0",
    protocol_version="1",
)

# A second registry in the same process, standing in for another package's.
other_registry = MigrationRegistry(
    protocol_name="other-proto",
    package_name="other",
    package_version="1.0.0",
    protocol_version="1",
)


class DeclaredInInitV1(MigratableObject, registry=covered_registry):
    canonical_name: str = "DeclaredInInit"
    version: str = "1"
