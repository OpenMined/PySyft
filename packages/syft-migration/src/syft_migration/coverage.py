"""Migration coverage checks a package runs against its own registry.

Two questions, asked the same way in every package: is every versioned object the
package defines registered in that package's registry, and can every registered
version reach every other version it has to reach.

Each check returns its findings instead of asserting, so a caller reports them
all at once. An empty list means the package is covered.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from syft_migration.base import MigratableObject
from syft_migration.identity import _has_identity, _identity, _version_order
from syft_migration.registry import MigrationRegistry


def _all_subclasses(cls: type) -> set[type]:
    subclasses = set(cls.__subclasses__())
    for sub in cls.__subclasses__():
        subclasses |= _all_subclasses(sub)
    return subclasses


def import_all_modules(package: ModuleType) -> None:
    """Import every submodule of ``package``.

    A versioned object registers when its class body runs, so a module nothing
    imports holds objects the registry has never seen.
    """
    for module_info in pkgutil.walk_packages(
        package.__path__, prefix=f"{package.__name__}."
    ):
        importlib.import_module(module_info.name)


def versioned_objects(package: ModuleType) -> list[type[MigratableObject]]:
    """Every concrete versioned object defined in ``package``, which it imports first.

    Abstract intermediates leave the identity fields required and are never
    registered, so they are not versioned objects and do not appear here.
    """
    import_all_modules(package)

    def defined_here(cls: type) -> bool:
        # The package's own __init__ module is named without the trailing dot,
        # so a class declared there needs the equality arm to be seen at all.
        return cls.__module__ == package.__name__ or cls.__module__.startswith(
            f"{package.__name__}."
        )

    return sorted(
        (
            cls
            for cls in _all_subclasses(MigratableObject)
            if defined_here(cls) and _has_identity(cls)
        ),
        key=_identity,
    )


def unregistered_objects(
    registry: MigrationRegistry, package: ModuleType
) -> list[type[MigratableObject]]:
    """Versioned objects in ``package`` that ``registry`` does not hold as themselves.

    ``MigratableObject`` already refuses a versioned class with no registry at
    all, so what is left to catch is a class registered into another package's
    registry: ``registry=`` is explicit, and a subclass inherits whichever one
    its parent named.
    """
    missing = []
    for cls in versioned_objects(package):
        canonical_name, version = _identity(cls)
        if registry.objects.get(canonical_name, {}).get(version) is not cls:
            missing.append(cls)
    return missing


def missing_upgrade_paths(registry: MigrationRegistry) -> list[tuple[str, str]]:
    """``(canonical_name, version)`` that cannot migrate up to the latest version."""
    return [
        (canonical_name, version)
        for canonical_name, versions in registry.objects.items()
        for version in versions
        if not registry.has_upgradeable_path_to_latest(
            canonical_name=canonical_name, from_version=version
        )
    ]


def missing_downgrade_paths(registry: MigrationRegistry) -> list[tuple[str, str, str]]:
    """``(canonical_name, from_version, to_version)`` with no downgrade between them.

    Versions order by the same integer key the registry uses, so version 10 sits
    above version 2 here even though a string sort puts it below.
    """
    missing = []
    for canonical_name, versions in registry.objects.items():
        ordered = sorted(versions, key=_version_order)
        for position, higher in enumerate(ordered):
            for lower in ordered[:position]:
                if not registry.has_migration_path(
                    canonical_name=canonical_name,
                    from_version=higher,
                    to_version=lower,
                ):
                    missing.append((canonical_name, higher, lower))
    return missing
