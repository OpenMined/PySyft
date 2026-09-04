from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from syft_migration.identity import MigrationError, _identity, _version_order

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject

PathLike = str | Path


class ProtocolSchema(BaseModel):
    """Every object version a registry supports for one protocol.

    ``protocol_name`` is a hardcoded, language-agnostic identifier for the
    protocol, intentionally distinct from any package name.
    """

    protocol_name: str
    # Incrementing protocol version ("0", "1", ...); bumped when the on-disk /
    # on-the-wire layout of the protocol changes, independent of package versions.
    version: str
    # The oldest protocol version this speaker still reads. A peer that predates
    # this field says nothing, so "0" refuses nothing.
    min_supported_version: str = "0"
    # canonical_name -> all supported versions
    supported_versions: dict[str, list[str]] = {}
    # canonical_name -> JSON schema of the protocol's current (latest) object
    # version; freezes what released classes look like so drift is detectable.
    current_object_schemas: dict[str, dict] = {}

    @classmethod
    def from_objects(
        cls,
        protocol_name: str,
        version: str,
        classes: list[type[MigratableObject]],
    ) -> ProtocolSchema:
        supported_versions: dict[str, list[str]] = {}
        latest_classes: dict[str, type[MigratableObject]] = {}
        for klass in classes:
            canonical_name, object_version = _identity(klass)
            versions = supported_versions.setdefault(canonical_name, [])
            if object_version not in versions:
                versions.append(object_version)
            if object_version == max(versions, key=_version_order):
                latest_classes[canonical_name] = klass
        return cls(
            protocol_name=protocol_name,
            version=version,
            supported_versions={
                name: sorted(versions, key=_version_order)
                for name, versions in supported_versions.items()
            },
            current_object_schemas={
                name: klass.model_json_schema()
                for name, klass in latest_classes.items()
            },
        )

    def current_schema(self, canonical_name: str) -> str:
        """The latest version this schema supports for ``canonical_name``."""
        versions = self.supported_versions.get(canonical_name)
        if not versions:
            raise MigrationError(f"Schema does not include object {canonical_name!r}")
        return max(versions, key=_version_order)

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> ProtocolSchema:
        return cls.model_validate(json.loads(Path(path).read_text()))


class PackageInfo(BaseModel):
    """Identity of one release of one package and the protocol version it speaks."""

    package_name: str
    version: str
    protocol_version: str


class ReleasedPackageProtocolInfo(BaseModel):
    """What EVERY package release emits: its identity + the protocol it speaks.

    Emitted also when the protocol did not change (the artifact then repeats the
    previous protocol version). Saved as a JSON release artifact so future
    releases can load the schemas of past releases (see
    MigrationRegistry.register_released_package_protocol_info).
    """

    package_info: PackageInfo
    protocol_schema: ProtocolSchema

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> ReleasedPackageProtocolInfo:
        return cls.model_validate(json.loads(Path(path).read_text()))


class ReleasedProtocol(BaseModel):
    """The frozen schema of one released protocol version.

    Emitted only by the release that actually CHANGED the protocol (the protocol
    version must only be bumped when the protocol changed).
    """

    protocol_schema: ProtocolSchema

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> ReleasedProtocol:
        return cls.model_validate(json.loads(Path(path).read_text()))
