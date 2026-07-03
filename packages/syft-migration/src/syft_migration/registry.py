from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable

from syft_migration.identity import MigrationError, _has_identity, _identity
from syft_migration.schema import PackageProtocolSchema, ProtocolSchema

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject

# A migration transforms one MigratableObject instance into another version.
MigrationFn = Callable[["MigratableObject"], "MigratableObject"]


class MigrationRegistry:
    """All known object versions, migrations, and protocol schemas for ONE package."""

    def __init__(
        self, protocol_name: str, package_name: str, package_version: str
    ) -> None:
        self.protocol_name = protocol_name
        self.package_name = package_name
        self.package_version = package_version
        # canonical_name -> {version: object_class}
        self.objects: dict[str, dict[str, type[MigratableObject]]] = {}
        # canonical_name -> {(from_version, to_version): migration_fn}
        self.migrations: dict[str, dict[tuple[str, str], MigrationFn]] = {}
        # package_version -> schema; usually read from files that are release
        # artifacts of earlier releases (see PackageProtocolSchema.save/load).
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

    def has_upgradeable_path_to_latest(
        self, canonical_name: str, from_version: str
    ) -> bool:
        """Whether ``from_version`` can be migrated up to the latest registered
        version of ``canonical_name`` (trivially true for the latest itself)."""
        try:
            self.migration_path(
                canonical_name=canonical_name,
                from_version=from_version,
                to_version=self.latest_version(canonical_name),
            )
        except MigrationError:
            return False
        return True

    # -- protocol schemas --------------------------------------------------
    @property
    def current_protocol_schema(self) -> PackageProtocolSchema:
        """The schema of THIS release, computed from the registered objects."""
        return PackageProtocolSchema(
            protocol_name=self.protocol_name,
            package_name=self.package_name,
            package_version=self.package_version,
            supported_versions={
                canonical_name: sorted(versions)
                for canonical_name, versions in self.objects.items()
            },
        )

    def register_historic_protocol_schema(self, schema: PackageProtocolSchema) -> None:
        """Register the schema of a PAST release of this package.

        Every object version the schema lists must already be registered (objects
        auto-register when their class is defined); registering a schema that
        references an unknown object/version raises before the schema is stored.
        """
        for canonical_name, versions in schema.supported_versions.items():
            for version in versions:
                self.get_class(canonical_name=canonical_name, version=version)
        self.history_protocol_schemas[schema.package_version] = schema

    def compute_protocol_schema(self) -> ProtocolSchema:
        """Every object version this registry can load, straight from ``self.objects``."""
        return ProtocolSchema(
            protocol_name=self.protocol_name,
            version=self.package_version,
            supported_versions={
                canonical_name: sorted(versions)
                for canonical_name, versions in self.objects.items()
            },
        )

    def schema_for_package_version(self, package_version: str) -> PackageProtocolSchema:
        if package_version == self.package_version:
            return self.current_protocol_schema
        try:
            return self.history_protocol_schemas[package_version]
        except KeyError:
            raise MigrationError(
                f"No protocol schema registered for package version {package_version!r}"
            )
