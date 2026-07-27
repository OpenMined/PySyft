from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from syft_migration.identity import MigrationError, _has_identity

if TYPE_CHECKING:
    from syft_migration.registry import MigrationRegistry


class MigratableObject(BaseModel):
    """Base class for any versioned object that can be migrated across versions.

    Subclasses pin their identity by overriding the field defaults, e.g.::

        class JobV2(MigratableObject, registry=my_registry):
            canonical_name: str = "job"
            version: str = "2"

    ``canonical_name`` is the stable logical name shared across all versions of the
    object; ``version`` is the schema version. Concrete subclasses auto-register into
    a :class:`MigrationRegistry` on definition. Pass ``registry=`` as a class keyword
    argument; subclasses of a registered class inherit its registry.
    """

    canonical_name: str
    version: str

    def __init_subclass__(
        cls, *, registry: Optional[MigrationRegistry] = None, **kwargs: object
    ) -> None:
        # Capture the chosen registry here; defer registration to
        # __pydantic_init_subclass__ where model_fields is fully built.
        if registry is not None:
            cls.__migration_registry__ = registry
        super().__init_subclass__(**kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        registry: Optional[MigrationRegistry] = getattr(
            cls, "__migration_registry__", None
        )
        if registry is None:
            if _has_identity(cls):
                raise MigrationError(
                    f"{cls.__name__} pins canonical_name/version but has no registry; "
                    "pass registry=... as a class keyword argument"
                )
            return
        registry.register_object_version(cls)
