from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from syft_migration.identity import MigrationError, _identity

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject

PathLike = str | Path


class BaseVersionsSchema(BaseModel):
    """Shared shape: all supported versions per object, keyed by canonical name."""

    # canonical_name -> all supported versions
    supported_versions: dict[str, list[str]] = {}

    def current_schema(self, canonical_name: str) -> str:
        """The latest version this schema supports for ``canonical_name``."""
        versions = self.supported_versions.get(canonical_name)
        if not versions:
            raise MigrationError(f"Schema does not include object {canonical_name!r}")
        return max(versions)


class ProtocolSchema(BaseVersionsSchema):
    """Every object version a registry supports for one protocol."""

    protocol_name: str
    # The package version that produced this schema.
    version: str


class PackageProtocolSchema(BaseVersionsSchema):
    """The protocol surface of one release of one package.

    Lists every object version that the package at ``package_version`` supports.
    ``protocol_name`` is a hardcoded, language-agnostic identifier for the protocol
    and is intentionally distinct from ``package_name``.
    """

    protocol_name: str
    package_name: str
    package_version: str

    @classmethod
    def from_objects(
        cls,
        protocol_name: str,
        package_name: str,
        package_version: str,
        classes: list[type[MigratableObject]],
    ) -> PackageProtocolSchema:
        supported_versions: dict[str, list[str]] = {}
        for klass in classes:
            canonical_name, version = _identity(klass)
            versions = supported_versions.setdefault(canonical_name, [])
            if version not in versions:
                versions.append(version)
        return cls(
            protocol_name=protocol_name,
            package_name=package_name,
            package_version=package_version,
            supported_versions={
                name: sorted(versions) for name, versions in supported_versions.items()
            },
        )

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> PackageProtocolSchema:
        return cls.model_validate(json.loads(Path(path).read_text()))
