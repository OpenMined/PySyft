# stdlib
from typing import Callable
from typing import Dict
from typing import Type

# relative
from ..util.autoreload import autoreload_enabled
from .syft_object import SyftObject


class SyftMigrationRegistry:
    __object_version_registry__: Dict[str, Dict[int : Type["SyftObject"]]] = {}
    __object_transform_registry__: Dict[str, Dict[str, Callable]] = {}

    @classmethod
    def register_version(cls, klass: Type[SyftObject]) -> None:
        klass = type(klass) if not isinstance(klass, type) else cls
        fqn = f"{klass.__module__}.{klass.__name__}"
        klass_version = klass.__version__

        if hasattr(klass, "__canonical_name__") and hasattr(klass, "__version__"):
            mapping_string = klass.__canonical_name__

            if (
                mapping_string in cls.__object_version_registry__
                and not autoreload_enabled()
            ):
                versions = klass.__object_version_registry__[mapping_string]
                versions[klass_version] = fqn
            else:
                # only if the cls has not been registered do we want to register it
                cls.__object_version_registry__[mapping_string] = {klass_version: fqn}

    @classmethod
    def register_transform(
        cls, klass_type: str, version_from: int, version_to: int, method: Callable
    ) -> None:
        if klass_type not in cls.__object_version_registry__:
            raise Exception(f"{klass_type} is not yet registered.")

        available_versions = cls.__object_version_registry__[klass_type]

        versions_exists = (
            version_from in available_versions and version_to in available_versions
        )

        if versions_exists:
            mapping_string = f"{version_from}x{version_to}"
            cls.__object_transform_registry__[klass_type][mapping_string] = method

        raise Exception(
            f"Available versions for {klass_type} are: {available_versions}."
            f"You're trying to add a transform from version: {version_from} to version: {version_to}"
        )
