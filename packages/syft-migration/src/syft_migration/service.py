from __future__ import annotations

from typing import Any, Optional

from syft_migration.base import MigratableObject
from syft_migration.identity import MigrationError
from syft_migration.registry import MigrationFn, MigrationRegistry
from syft_migration.schema import PackageProtocolSchema, ProtocolSchema


class MigrationService:
    """Upgrades and downgrades migratable objects using a package's registry."""

    def __init__(self, registry: MigrationRegistry) -> None:
        self.registry = registry

    def migrate(self, obj: MigratableObject, target_version: str) -> MigratableObject:
        """Migrate ``obj`` to ``target_version`` by applying the registered path."""
        path: list[MigrationFn] = self.registry.migration_path(
            obj.canonical_name, obj.version, target_version
        )
        result = obj
        for migration in path:
            result = migration(result)
        return result

    def migrate_to_schema(
        self, obj: MigratableObject, schema: PackageProtocolSchema
    ) -> MigratableObject:
        """Migrate ``obj`` to the version pinned by ``schema`` (the on-the-fly downgrade)."""
        target_version = schema.object_versions.get(obj.canonical_name)
        if target_version is None:
            raise MigrationError(
                f"Protocol schema {schema.package_name}@{schema.package_version} does "
                f"not include object {obj.canonical_name!r}"
            )
        return self.migrate(obj, target_version)

    def downgrade_for_package_version(
        self, obj: MigratableObject, package_version: str
    ) -> MigratableObject:
        """Migrate ``obj`` to the version a peer running ``package_version`` understands."""
        schema = self.registry.schema_for_package_version(package_version)
        return self.migrate_to_schema(obj, schema)

    def export_protocol_schema(self) -> ProtocolSchema:
        """Export every object version this package supports."""
        return self.registry.compute_protocol_schema()

    def load(
        self, data: dict[str, Any], target_version: Optional[str] = None
    ) -> MigratableObject:
        """Deserialize ``data`` into its historical class, optionally migrating it."""
        try:
            canonical_name = data["canonical_name"]
            version = data["version"]
        except KeyError as exc:
            raise MigrationError(f"Serialized object is missing {exc} field")
        cls = self.registry.get_class(canonical_name, version)
        obj = cls.model_validate(data)
        if target_version is not None:
            return self.migrate(obj, target_version)
        return obj
