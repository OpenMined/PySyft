from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syft_migration.base import MigratableObject


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


def _version_order(version: str) -> int:
    """Return the sort key of an object version.

    Object versions are incrementing integers held as strings. A string sort puts
    ``"10"`` before ``"2"``, so every comparison must use this key.
    """
    try:
        return int(version)
    except ValueError:
        raise MigrationError(
            f"Object version {version!r} is not an integer. Object versions are "
            "incrementing integers, for example '1', '2', '3'."
        ) from None


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
