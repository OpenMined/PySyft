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


def _has_identity(cls: type[MigratableObject]) -> bool:
    """Whether ``cls`` pins both identity fields (i.e. is a concrete version).

    The base class and abstract intermediates leave the fields required (no
    default), so they have no identity and are not registered.
    """
    name_field = cls.model_fields.get("canonical_name")
    version_field = cls.model_fields.get("version")
    if name_field is None or version_field is None:
        return False
    return not (name_field.is_required() or version_field.is_required())


def _identity(cls: type[MigratableObject]) -> tuple[str, str]:
    """Return (canonical_name, version) for a concrete subclass.

    Raises ``MigrationError`` if ``cls`` does not pin both fields (the base class
    and abstract intermediates leave them required, so they have no identity).
    """
    if not _has_identity(cls):
        raise MigrationError(
            f"{cls.__name__} does not pin canonical_name/version and has no identity"
        )
    return (
        str(cls.model_fields["canonical_name"].default),
        str(cls.model_fields["version"].default),
    )


class MigrationRegistry:
    """All known object versions, migrations, and protocol schemas for ONE package."""

    def __init__(self) -> None:
        # canonical_name -> {version: object_class}
        self.objects: dict[str, dict[str, type[MigratableObject]]] = {}
        # canonical_name -> {(from_version, to_version): migration_fn}
        self.migrations: dict[str, dict[tuple[str, str], MigrationFn]] = {}
        self.current_protocol_schema: Optional[PackageProtocolSchema] = None
        self.history_protocol_schemas: dict[str, PackageProtocolSchema] = {}

    # -- objects -----------------------------------------------------------
    def register_object_version(self, cls: type[MigratableObject]) -> None:
        if not _has_identity(cls):
            return
        canonical_name, version = _identity(cls)
        existing = self.objects.get(canonical_name, {}).get(version)
        if existing is not None and existing is not cls:
            raise MigrationError(
                f"Object {(canonical_name, version)} already registered as "
                f"{existing.__name__}, cannot re-register as {cls.__name__}"
            )
        self.objects.setdefault(canonical_name, {})[version] = cls

    def get_class(self, canonical_name: str, version: str) -> type[MigratableObject]:
        try:
            return self.objects[canonical_name][version]
        except KeyError:
            raise MigrationError(
                f"No object registered for {(canonical_name, version)}"
            )

    def versions(self, canonical_name: str) -> list[str]:
        return list(self.objects.get(canonical_name, {}))

    def latest_version(self, canonical_name: str) -> str:
        schema = self.current_protocol_schema
        if schema is not None and canonical_name in schema.object_versions:
            return schema.object_versions[canonical_name]
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
        fn: MigrationFn,
    ) -> None:
        """Register ``fn`` as the migration from ``from_version`` to ``to_version``."""
        edges = self.migrations.setdefault(canonical_name, {})
        edges[(from_version, to_version)] = fn

    def migration(
        self, canonical_name: str, from_version: str, to_version: str
    ) -> Callable[[MigrationFn], MigrationFn]:
        """Decorator form of :meth:`register_migration` for named functions."""

        def decorator(fn: MigrationFn) -> MigrationFn:
            self.register_migration(
                canonical_name=canonical_name,
                from_version=from_version,
                to_version=to_version,
                fn=fn,
            )
            return fn

        return decorator

    def migration_path(
        self, canonical_name: str, from_version: str, to_version: str
    ) -> list[MigrationFn]:
        """Return the migration functions to apply, in order, via BFS over edges."""
        if from_version == to_version:
            return []
        # all migrations for this class
        edges = self.migrations.get(canonical_name, {})
        # BFS queue of (current_version, path_to_current)
        queue: deque[tuple[str, list[MigrationFn]]] = deque([(from_version, [])])
        # seen versions
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

        Every object the schema pins must already be registered (objects auto-register
        when their class is defined); registering a schema that references an unknown
        object/version raises before the schema is stored.
        """
        for canonical_name, version in schema.object_versions.items():
            self.get_class(canonical_name=canonical_name, version=version)
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
