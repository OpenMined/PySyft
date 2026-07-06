from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from syft_migration.identity import MigrationError, _identity

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
    # canonical_name -> all supported versions
    supported_versions: dict[str, list[str]] = {}

    @classmethod
    def from_objects(
        cls,
        protocol_name: str,
        version: str,
        classes: list[type[MigratableObject]],
    ) -> ProtocolSchema:
        supported_versions: dict[str, list[str]] = {}
        for klass in classes:
            canonical_name, object_version = _identity(klass)
            versions = supported_versions.setdefault(canonical_name, [])
            if object_version not in versions:
                versions.append(object_version)
        return cls(
            protocol_name=protocol_name,
            version=version,
            supported_versions={
                name: sorted(versions) for name, versions in supported_versions.items()
            },
        )

    def current_schema(self, canonical_name: str) -> str:
        """The latest version this schema supports for ``canonical_name``."""
        versions = self.supported_versions.get(canonical_name)
        if not versions:
            raise MigrationError(f"Schema does not include object {canonical_name!r}")
        return max(versions)

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> ProtocolSchema:
        return cls.model_validate(json.loads(Path(path).read_text()))


class PackageInfo(BaseModel):
    """Identity of one release of one package."""

    package_name: str
    version: str


class ReleaseArtifact(BaseModel):
    """What one release emits: the package's identity + the protocol schema it speaks.

    Saved as a JSON release artifact so future releases can load the schemas of
    past releases (see MigrationRegistry.register_historic_release_artifact).
    """

    package_info: PackageInfo
    protocol_schema: ProtocolSchema

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> ReleaseArtifact:
        return cls.model_validate(json.loads(Path(path).read_text()))
