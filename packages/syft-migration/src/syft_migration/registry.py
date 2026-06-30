from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject
    from syft_migration.schema import PackageProtocolSchema

# A migration transforms one MigratableObject instance into another version.
MigrationFn = Callable[["MigratableObject"], "MigratableObject"]


class MigrationError(Exception):
    """Raised when an object cannot be registered, located, or migrated."""


def _identity(cls: type[MigratableObject]) -> Optional[tuple[str, str]]:
    """Return (canonical_name, version) for a concrete subclass, else None.

    The base class and abstract intermediates leave the fields required (no
    default), so they have no identity and are not registered.
    """
    name_field = cls.model_fields.get("canonical_name")
    version_field = cls.model_fields.get("version")
    if name_field is None or version_field is None:
        return None
    if name_field.is_required() or version_field.is_required():
        return None
    return str(name_field.default), str(version_field.default)


class MigrationRegistry:
    """All known object versions, migrations, and protocol schemas for ONE package."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], type[MigratableObject]] = {}
        self.migrations: dict[str, dict[tuple[str, str], MigrationFn]] = {}
        self.current_protocol_schema: Optional[PackageProtocolSchema] = None
        self.history_protocol_schemas: dict[str, PackageProtocolSchema] = {}

    # -- objects -----------------------------------------------------------
    def register_object(self, cls: type[MigratableObject]) -> None:
        identity = _identity(cls)
        if identity is None:
            return
        existing = self.objects.get(identity)
        if existing is not None and existing is not cls:
            raise MigrationError(
                f"Object {identity} already registered as {existing.__name__}, "
                f"cannot re-register as {cls.__name__}"
            )
        self.objects[identity] = cls

    def get_class(self, canonical_name: str, version: str) -> type[MigratableObject]:
        try:
            return self.objects[(canonical_name, version)]
        except KeyError:
            raise MigrationError(
                f"No object registered for {(canonical_name, version)}"
            )

    def versions(self, canonical_name: str) -> list[str]:
        return [v for (name, v) in self.objects if name == canonical_name]

    def latest_version(self, canonical_name: str) -> str:
        schema = self.current_protocol_schema
        if schema is not None and canonical_name in schema.objects:
            return schema.objects[canonical_name]
        versions = self.versions(canonical_name)
        if not versions:
            raise MigrationError(f"No versions registered for {canonical_name!r}")
        return max(versions)

    # -- migrations --------------------------------------------------------
    def register_migration(
        self,
        canonical_name: str,
        from_version: str,
        to_version: str,
        fn: Optional[MigrationFn] = None,
    ) -> Callable[[MigrationFn], MigrationFn] | None:
        edges = self.migrations.setdefault(canonical_name, {})

        def _add(func: MigrationFn) -> MigrationFn:
            edges[(from_version, to_version)] = func
            return func

        if fn is None:
            return _add
        _add(fn)
        return None

    def migration_path(
        self, canonical_name: str, from_version: str, to_version: str
    ) -> list[MigrationFn]:
        """Return the migration functions to apply, in order, via BFS over edges."""
        if from_version == to_version:
            return []
        edges = self.migrations.get(canonical_name, {})
        queue: deque[tuple[str, list[MigrationFn]]] = deque([(from_version, [])])
        seen = {from_version}
        while queue:
            current, path = queue.popleft()
            for (src, dst), func in edges.items():
                if src != current or dst in seen:
                    continue
                next_path = [*path, func]
                if dst == to_version:
                    return next_path
                seen.add(dst)
                queue.append((dst, next_path))
        raise MigrationError(
            f"No migration path for {canonical_name!r} from {from_version} to {to_version}"
        )

    # -- protocol schemas --------------------------------------------------
    def register_protocol_schema(
        self, schema: PackageProtocolSchema, *, current: bool = True
    ) -> None:
        """Register a schema, keeping the object registry and schemas in sync.

        Every object the schema ships is registered first (added if absent, raising
        on a conflicting class), then the schema is stored.
        """
        for cls in schema.object_classes():
            self.register_object(cls)
        self.history_protocol_schemas[schema.package_version] = schema
        if current:
            self.current_protocol_schema = schema

    def schema_for_package_version(self, package_version: str) -> PackageProtocolSchema:
        if (
            self.current_protocol_schema is not None
            and self.current_protocol_schema.package_version == package_version
        ):
            return self.current_protocol_schema
        try:
            return self.history_protocol_schemas[package_version]
        except KeyError:
            raise MigrationError(
                f"No protocol schema registered for package version {package_version!r}"
            )


# Default per-import registry used by MigratableObject.__init_subclass__.
default_registry = MigrationRegistry()
