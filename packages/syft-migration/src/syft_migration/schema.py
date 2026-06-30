from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, PrivateAttr

from syft_migration.registry import MigrationError, _identity

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject

PathLike = str | Path


class PackageProtocolSchema(BaseModel):
    """The protocol surface of one release of one package.

    Pins exactly one ``version`` per object (``canonical_name``) that the package ships
    at ``package_version``. ``protocol_name`` is a hardcoded, language-agnostic
    identifier for the protocol and is intentionally distinct from ``package_name``.
    """

    protocol_name: str
    package_name: str
    package_version: str
    objects: dict[str, str] = {}

    # Class references for the pinned objects. Not serialized; only present on schemas
    # built in code (a schema loaded from metadata has the version map but no classes).
    _object_classes: dict[str, type[MigratableObject]] = PrivateAttr(
        default_factory=dict
    )

    @classmethod
    def from_objects(
        cls,
        protocol_name: str,
        package_name: str,
        package_version: str,
        classes: list[type[MigratableObject]],
    ) -> PackageProtocolSchema:
        objects: dict[str, str] = {}
        object_classes: dict[str, type[MigratableObject]] = {}
        for klass in classes:
            identity = _identity(klass)
            if identity is None:
                raise MigrationError(
                    f"{klass.__name__} has no canonical_name/version and cannot be "
                    "added to a protocol schema"
                )
            canonical_name, version = identity
            if canonical_name in objects:
                raise MigrationError(
                    f"Protocol schema may only pin one version per object, but "
                    f"{canonical_name!r} was given twice"
                )
            objects[canonical_name] = version
            object_classes[canonical_name] = klass
        schema = cls(
            protocol_name=protocol_name,
            package_name=package_name,
            package_version=package_version,
            objects=objects,
        )
        schema._object_classes = object_classes
        return schema

    def object_classes(self) -> list[type[MigratableObject]]:
        return list(self._object_classes.values())

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: PathLike) -> PackageProtocolSchema:
        return cls.model_validate(json.loads(Path(path).read_text()))
