from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable

from syft_migration.identity import (
    MigrationError,
    _has_identity,
    _identity,
    _version_order,
)
from syft_migration.schema import (
    PackageInfo,
    ProtocolSchema,
    ReleasedPackageProtocolInfo,
    ReleasedProtocol,
)

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject

# A migration transforms one MigratableObject instance into another version.
MigrationFn = Callable[["MigratableObject"], "MigratableObject"]


class MigrationRegistry:
    """All known object versions, migrations, and protocol schemas for ONE package."""

    def __init__(
        self,
        protocol_name: str,
        package_name: str,
        package_version: str,
        protocol_version: str,
        min_supported_protocol_version: str = "0",
    ) -> None:
        self.protocol_name = protocol_name
        self.package_name = package_name
        self.package_version = package_version
        self.protocol_version = protocol_version
        # The oldest protocol version this package still reads. Raise it only
        # when the code drops support for a protocol that a release froze.
        self.min_supported_protocol_version = min_supported_protocol_version
        # canonical_name -> {version: object_class}
        self.objects: dict[str, dict[str, type[MigratableObject]]] = {}
        # canonical_name -> {(from_version, to_version): migration_fn}
        self.migrations: dict[str, dict[tuple[str, str], MigrationFn]] = {}

        # protocol_version -> identity of the past release that spoke that
        # protocol; usually read from files that are release artifacts of
        # earlier releases (see ReleasedPackageProtocolInfo.save/load).
        self.package_version_history: dict[str, PackageInfo] = {}
        # protocol_version -> schema of the release that spoke that protocol
        self.protocol_version_history: dict[str, ProtocolSchema] = {}

    # -- objects -----------------------------------------------------------
    def register_object_version(self, cls: type[MigratableObject]) -> None:
        if not _has_identity(cls):
            return
        canonical_name, version = _identity(cls)
        # Reject a version that cannot be ordered, at class definition time.
        _version_order(version)
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
        return max(versions, key=_version_order)

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

    def has_migration_path(
        self, canonical_name: str, from_version: str, to_version: str
    ) -> bool:
        """Whether a chain of migrations leads from ``from_version`` to ``to_version``."""
        try:
            self.migration_path(
                canonical_name=canonical_name,
                from_version=from_version,
                to_version=to_version,
            )
        except MigrationError:
            return False
        return True

    def has_upgradeable_path_to_latest(
        self, canonical_name: str, from_version: str
    ) -> bool:
        """Whether ``from_version`` can be migrated up to the latest registered
        version of ``canonical_name`` (trivially true for the latest itself)."""
        try:
            latest = self.latest_version(canonical_name)
        except MigrationError:
            return False
        return self.has_migration_path(canonical_name, from_version, latest)

    # -- protocol schemas --------------------------------------------------
    def _raise_for_unknown_objects(self, schema: ProtocolSchema) -> None:
        """Raise if the schema lists an object version this registry cannot load."""
        for canonical_name, versions in schema.supported_versions.items():
            for version in versions:
                self.get_class(canonical_name=canonical_name, version=version)

    def register_historic_protocol_schema(
        self, schema: ProtocolSchema, raise_for_unknown_objects: bool = False
    ) -> None:
        """Register the schema of a PAST protocol version of this package.

        With ``raise_for_unknown_objects`` every object version the schema lists
        must already be registered (objects auto-register when their class is
        defined); a schema referencing an unknown object/version then raises
        before the schema is stored.
        """
        if raise_for_unknown_objects:
            self._raise_for_unknown_objects(schema)
        self.protocol_version_history[schema.version] = schema

    def register_released_package_protocol_info(
        self,
        info: ReleasedPackageProtocolInfo,
        raise_for_unknown_objects: bool = False,
    ) -> None:
        """Register the release info of a PAST release of this package.

        Registers the release's protocol schema and remembers which package
        release spoke that protocol version.
        """
        self.register_historic_protocol_schema(
            schema=info.protocol_schema,
            raise_for_unknown_objects=raise_for_unknown_objects,
        )
        protocol_version = info.package_info.protocol_version
        self.package_version_history[protocol_version] = info.package_info

    def register_released_protocol(
        self, released: ReleasedProtocol, raise_for_unknown_objects: bool = False
    ) -> None:
        """Register the frozen schema of a PAST released protocol version."""
        self.register_historic_protocol_schema(
            schema=released.protocol_schema,
            raise_for_unknown_objects=raise_for_unknown_objects,
        )

    def compute_protocol_schema(self) -> ProtocolSchema:
        """Every object version this registry can load, straight from ``self.objects``."""
        return ProtocolSchema(
            protocol_name=self.protocol_name,
            version=self.protocol_version,
            min_supported_version=self.min_supported_protocol_version,
            supported_versions={
                canonical_name: sorted(versions, key=_version_order)
                for canonical_name, versions in self.objects.items()
            },
            current_object_schemas={
                canonical_name: self.get_class(
                    canonical_name, self.latest_version(canonical_name)
                ).model_json_schema()
                for canonical_name in self.objects
            },
        )

    def negotiate_protocol_version(
        self, peer_version: str, peer_min: str | None = None
    ) -> str:
        """The protocol version to speak with a peer.

        Both sides speak the lower of the two current versions, because each side
        must read what the other writes. That version must also be at or above
        both floors. A peer that publishes no floor is treated as ``"0"``, which
        refuses nothing.

        Raises MigrationError when no version satisfies both sides.
        """
        chosen = min(self.protocol_version, peer_version, key=_version_order)
        floor = max(
            self.min_supported_protocol_version, peer_min or "0", key=_version_order
        )
        if _version_order(chosen) < _version_order(floor):
            raise MigrationError(
                f"No usable {self.protocol_name} protocol version with this peer. "
                f"This client speaks {self.protocol_version} and reads down to "
                f"{self.min_supported_protocol_version}; the peer speaks "
                f"{peer_version} and reads down to {peer_min or '0'}."
            )
        return chosen

    def compute_released_protocol(self) -> ReleasedProtocol:
        """The protocol artifact a release emits when the protocol changed."""
        return ReleasedProtocol(protocol_schema=self.compute_protocol_schema())

    def compute_released_package_protocol_info(self) -> ReleasedPackageProtocolInfo:
        """The artifact EVERY package release emits."""
        return ReleasedPackageProtocolInfo(
            package_info=PackageInfo(
                package_name=self.package_name,
                version=self.package_version,
                protocol_version=self.protocol_version,
            ),
            protocol_schema=self.compute_protocol_schema(),
        )

    def schema_for_protocol_version(self, protocol_version: str) -> ProtocolSchema:
        if protocol_version == self.protocol_version:
            return self.compute_protocol_schema()
        try:
            return self.protocol_version_history[protocol_version]
        except KeyError:
            raise MigrationError(
                f"No protocol schema registered for protocol version {protocol_version!r}"
            )

    # -- released schema integrity ------------------------------------------
    def find_schema_drift(self) -> list[tuple[str, str, str]]:
        """Check the current registry against ALL released protocols in history.

        A released protocol froze, per canonical name, the JSON schema of its
        current object version; the class registered for that version must still
        produce the same schema. Returns the drifted
        (canonical_name, object_version, protocol_version) tuples; empty = clean.
        """
        drifted = []
        for protocol_version, schema in self.protocol_version_history.items():
            for canonical_name, frozen in schema.current_object_schemas.items():
                version = schema.current_schema(canonical_name)
                try:
                    current = self.get_class(
                        canonical_name, version
                    ).model_json_schema()
                except MigrationError:
                    current = None
                if current != frozen:
                    drifted.append((canonical_name, version, protocol_version))
        return drifted

    def protocol_changed_without_bump(self) -> bool:
        """Whether the protocol changed while its version constant did not.

        A package release must only bump the protocol version when the protocol
        actually changed — and must bump it when it did. If a released protocol
        schema exists for THIS registry's protocol version, its supported
        versions must match the ones computed from the code.
        """
        released = self.protocol_version_history.get(self.protocol_version)
        if released is None:
            return False
        current = self.compute_protocol_schema()
        return released.supported_versions != current.supported_versions

    def latest_released_protocol_version(self) -> str | None:
        """The newest protocol version with a frozen schema. None if there is none."""
        if not self.protocol_version_history:
            return None
        return max(self.protocol_version_history, key=_version_order)

    def protocol_bump_missing(self) -> bool:
        """Whether the protocol changed since the newest RELEASED protocol
        without a bump of the version constant.

        Only object versions are compared. A protocol change that alters the
        on-disk layout, but adds no object version, is invisible here.
        """
        latest = self.latest_released_protocol_version()
        if latest is None:
            return False
        released = self.protocol_version_history[latest]
        current = self.compute_protocol_schema()
        if current.supported_versions == released.supported_versions:
            return False
        return _version_order(self.protocol_version) <= _version_order(latest)
