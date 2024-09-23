# stdlib
from collections.abc import Callable
from typing import Any
from typing import TYPE_CHECKING

SYFT_086_PROTOCOL_VERSION = "4"

# third party

# relative
if TYPE_CHECKING:
    # relative
    from .syft_object import SyftObject


class SyftObjectRegistry:
    __object_transform_registry__: dict[str, Callable] = {}
    __object_serialization_registry__: dict[str, dict[int, tuple]] = {}
    __type_to_canonical_name__: dict[type, tuple[str, int]] = {}

    @classmethod
    def register_cls(
        cls, canonical_name: str, version: int, serde_attributes: tuple
    ) -> None:
        if canonical_name not in cls.__object_serialization_registry__:
            cls.__object_serialization_registry__[canonical_name] = {}
        cls.__object_serialization_registry__[canonical_name][version] = (
            serde_attributes
        )

        cls.__type_to_canonical_name__[serde_attributes[7]] = (canonical_name, version)

    @classmethod
    def get_versions(cls, canonical_name: str) -> list[int]:
        available_versions: dict = cls.__object_serialization_registry__.get(
            canonical_name,
            {},
        )
        return list(available_versions.keys())

    @classmethod
    def get_latest_version(cls, canonical_name: str) -> int:
        available_versions = cls.get_versions(canonical_name)
        if not available_versions:
            return 0
        return sorted(available_versions, reverse=True)[0]

    @classmethod
    def get_identifier_for_type(cls, obj: Any) -> tuple[str, int]:
        """
        This is to create the string in nonrecursiveBlob
        """
        return cls.__type_to_canonical_name__[obj]

    @classmethod
    def get_canonical_name_version(cls, obj: Any) -> tuple[str, int]:
        """
        Retrieves the canonical name for both objects and types.

        This function works for both objects and types, returning the canonical name
        as a string. It handles various cases, including built-in types, instances of
        classes, and enum members.

        If the object is not registered in the registry, a ValueError is raised.

        Examples:
            get_canonical_name_version([1,2,3]) -> "list"
            get_canonical_name_version(list) -> "type"
            get_canonical_name_version(MyEnum.A) -> "MyEnum"
            get_canonical_name_version(MyEnum) -> "type"

        Args:
            obj: The object or type for which to get the canonical name.

        Returns:
            The canonical name and version of the object or type.
        """

        # for types we return "type"
        if isinstance(obj, type):
            return cls.__type_to_canonical_name__[type]

        obj_type = type(obj)
        if obj_type in cls.__type_to_canonical_name__:
            return cls.__type_to_canonical_name__[obj_type]

        raise ValueError(
            f"Could not find canonical name for '{obj_type.__module__}.{obj_type.__name__}'"
        )

    @classmethod
    def get_serde_properties(cls, canonical_name: str, version: int) -> tuple:
        try:
            return cls.__object_serialization_registry__[canonical_name][version]
        except Exception:
            # This is a hack for python 3.10 in which Any is not a type
            # if the server uses py>3.10 and the client 3.10 this goes wrong
            if canonical_name == "Any_typing._SpecialForm":
                return cls.__object_serialization_registry__["Any"][version]
            else:
                if canonical_name not in cls.__object_serialization_registry__:
                    raise ValueError(f"Could not find {canonical_name} in registry")
                elif (
                    version not in cls.__object_serialization_registry__[canonical_name]
                ):
                    raise ValueError(
                        f"Could not find {canonical_name} version {version} in registry"
                    )
                else:
                    raise

    @classmethod
    def get_serde_class(cls, canonical_name: str, version: int) -> type["SyftObject"]:
        serde_properties = cls.get_serde_properties(canonical_name, version)
        return serde_properties[7]

    @classmethod
    def has_serde_class(cls, canonical_name: str | None, version: int) -> bool:
        # relative
        return (
            canonical_name in cls.__object_serialization_registry__
            and version in cls.__object_serialization_registry__[canonical_name]
        )

    @classmethod
    def add_transform(
        cls,
        klass_from: str,
        version_from: int,
        klass_to: str,
        version_to: int,
        method: Callable,
    ) -> None:
        mapping_string = f"{klass_from}_{version_from}_x_{klass_to}_{version_to}"
        cls.__object_transform_registry__[mapping_string] = method

    @classmethod
    def get_transform(
        cls, type_from: type["SyftObject"], type_to: type["SyftObject"]
    ) -> Callable:
        # relative
        from .syft_object import SyftBaseObject
        from .syft_object import SyftObject

        for type_from_mro in type_from.mro():
            if issubclass(type_from_mro, SyftObject):
                klass_from = type_from_mro.__canonical_name__
                version_from = type_from_mro.__version__
            else:
                klass_from = type_from_mro.__name__
                version_from = None
            for type_to_mro in type_to.mro():
                if (
                    issubclass(type_to_mro, SyftBaseObject)
                    and type_to_mro != SyftBaseObject
                ):
                    klass_to = type_to_mro.__canonical_name__
                    version_to = type_to_mro.__version__
                else:
                    klass_to = type_to_mro.__name__
                    version_to = None

                mapping_string = (
                    f"{klass_from}_{version_from}_x_{klass_to}_{version_to}"
                )
                if mapping_string in SyftObjectRegistry.__object_transform_registry__:
                    return SyftObjectRegistry.__object_transform_registry__[
                        mapping_string
                    ]
        raise Exception(
            f"No mapping found for: {type_from} to {type_to} in"
            f"the registry: {SyftObjectRegistry.__object_transform_registry__.keys()}"
        )
