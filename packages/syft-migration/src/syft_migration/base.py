from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from syft_migration.registry import MigrationRegistry, default_registry


class MigratableObject(BaseModel):
    """Base class for any versioned object that can be migrated across versions.

    Subclasses pin their identity by overriding the field defaults, e.g.::

        class JobV2(MigratableObject):
            canonical_name: str = "job"
            version: str = "2"

    ``canonical_name`` is the stable logical name shared across all versions of the
    object; ``version`` is the schema version. Concrete subclasses auto-register into
    a :class:`MigrationRegistry` on definition. Pass ``registry=`` as a class keyword
    argument to register into a non-default registry (handy for test isolation).
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
        registry: MigrationRegistry = getattr(
            cls, "__migration_registry__", default_registry
        )
        registry.register_object_version(cls)
