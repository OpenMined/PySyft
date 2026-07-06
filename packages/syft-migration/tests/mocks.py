from syft_migration import (
    MigratableObject,
    MigrationRegistry,
    PackageInfo,
    ProtocolSchema,
    ReleaseArtifact,
)

# Isolated registry for the mock objects.
mock_registry = MigrationRegistry(
    protocol_name="mock-proto",
    package_name="syft-mock",
    package_version="1.1.0",
    protocol_version="2",
)


class JobV1(MigratableObject, registry=mock_registry):
    canonical_name: str = "job"
    version: str = "1"
    name: str


class JobV2(MigratableObject, registry=mock_registry):
    canonical_name: str = "job"
    version: str = "2"
    name: str
    owner: str = "unknown"


class JobV3(MigratableObject, registry=mock_registry):
    canonical_name: str = "job"
    version: str = "3"
    name: str
    owner: str = "unknown"
    priority: int = 0


mock_registry.register_migration(
    canonical_name="job",
    from_version="1",
    to_version="2",
    fn=lambda obj: JobV2(name=obj.name),
)
mock_registry.register_migration(
    canonical_name="job",
    from_version="2",
    to_version="3",
    fn=lambda obj: JobV3(name=obj.name, owner=obj.owner),
)
mock_registry.register_migration(
    canonical_name="job",
    from_version="2",
    to_version="1",
    fn=lambda obj: JobV1(name=obj.name),
)

# The release artifact of an earlier release; the current schema is computed
# by the registry.
schema_v1 = ProtocolSchema.from_objects(
    protocol_name="mock-proto",
    version="1",
    classes=[JobV1],
)
mock_registry.register_historic_release_artifact(
    artifact=ReleaseArtifact(
        package_info=PackageInfo(
            package_name="syft-mock", version="1.0.0", protocol_version="1"
        ),
        protocol_schema=schema_v1,
    )
)
