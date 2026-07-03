from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from syft_migration.registry import MigrationError, _identity

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject

PathLike = str | Path


class ProtocolSchema(BaseModel):
    """Every object version a registry supports for one protocol.

    Unlike :class:`PackageProtocolSchema` (which pins exactly one version per object
    for one package release), this lists ALL versions the running package can load
    and migrate, keyed by canonical name.
    """

    protocol_name: str
    # The package version that produced this schema.
    version: str
    # canonical_name -> all supported versions
    supported_versions: dict[str, list[str]] = {}


class PackageProtocolSchema(BaseModel):
    """The protocol surface of one release of one package.

    Pins exactly one ``version`` per object (``canonical_name``) that the package ships
    at ``package_version``. ``protocol_name`` is a hardcoded, language-agnostic
    identifier for the protocol and is intentionally distinct from ``package_name``.
    """

    protocol_name: str
    package_name: str
    package_version: str
    # canonical_name -> version
    object_versions: dict[str, str] = {}

    @classmethod
    def from_objects(
        cls,
        protocol_name: str,
        package_name: str,
        package_version: str,
        classes: list[type[MigratableObject]],
    ) -> PackageProtocolSchema:
        object_versions: dict[str, str] = {}
        for klass in classes:
            canonical_name, version = _identity(klass)
            if canonical_name in object_versions:
                raise MigrationError(
                    f"Protocol schema may only pin one version per object, but "
                    f"{canonical_name!r} was given twice"
                )
            object_versions[canonical_name] = version
        return cls(
            protocol_name=protocol_name,
            package_name=package_name,
            package_version=package_version,
            object_versions=object_versions,
        )

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> PackageProtocolSchema:
        return cls.model_validate(json.loads(Path(path).read_text()))
