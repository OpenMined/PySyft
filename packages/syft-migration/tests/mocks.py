from syft_migration import (
    MigratableObject,
    MigrationRegistry,
    PackageProtocolSchema,
)

# Isolated registry so the mock objects never touch the global default_registry.
mock_registry = MigrationRegistry()


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


mock_registry.register_migration("job", "1", "2", lambda obj: JobV2(name=obj.name))
mock_registry.register_migration(
    "job", "2", "3", lambda obj: JobV3(name=obj.name, owner=obj.owner)
)
mock_registry.register_migration("job", "2", "1", lambda obj: JobV1(name=obj.name))

schema_v1 = PackageProtocolSchema.from_objects(
    "mock-proto", "syft-mock", "1.0.0", [JobV1]
)
schema_v2 = PackageProtocolSchema.from_objects(
    "mock-proto", "syft-mock", "1.1.0", [JobV2]
)
mock_registry.register_protocol_schema(schema_v1, current=False)
mock_registry.register_protocol_schema(schema_v2, current=True)
