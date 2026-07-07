from syft_migration import (
    MigratableObject,
    MigrationRegistry,
    PackageInfo,
    ProtocolSchema,
    ReleasedPackageProtocolInfo,
)

# Isolated registry for the mock objects.
mock_registry = MigrationRegistry(
    protocol_name="mock-proto",
    package_name="syft-mock",
    package_version="1.1.0",
    protocol_version="2",
)


class MyVersionedObjectV1(MigratableObject, registry=mock_registry):
    canonical_name: str = "MyVersionedObject"
    version: str = "1"
    name: str


class MyVersionedObjectV2(MigratableObject, registry=mock_registry):
    canonical_name: str = "MyVersionedObject"
    version: str = "2"
    name: str
    owner: str = "unknown"


class MyVersionedObjectV3(MigratableObject, registry=mock_registry):
    canonical_name: str = "MyVersionedObject"
    version: str = "3"
    name: str
    owner: str = "unknown"
    priority: int = 0


mock_registry.register_migration(
    canonical_name="MyVersionedObject",
    from_version="1",
    to_version="2",
    fn=lambda obj: MyVersionedObjectV2(name=obj.name),
)
mock_registry.register_migration(
    canonical_name="MyVersionedObject",
    from_version="2",
    to_version="3",
    fn=lambda obj: MyVersionedObjectV3(name=obj.name, owner=obj.owner),
)
mock_registry.register_migration(
    canonical_name="MyVersionedObject",
    from_version="2",
    to_version="1",
    fn=lambda obj: MyVersionedObjectV1(name=obj.name),
)

# The release artifact of an earlier release; the current schema is computed
# by the registry.
schema_v1 = ProtocolSchema.from_objects(
    protocol_name="mock-proto",
    version="1",
    classes=[MyVersionedObjectV1],
)
mock_registry.register_released_package_protocol_info(
    info=ReleasedPackageProtocolInfo(
        package_info=PackageInfo(
            package_name="syft-mock", version="1.0.0", protocol_version="1"
        ),
        protocol_schema=schema_v1,
    )
)
