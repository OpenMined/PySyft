# stdlib
from collections.abc import Callable
from typing import Any
from typing import TYPE_CHECKING

# relative
from ..util.util import get_fully_qualified_name

SYFT_086_PROTOCOL_VERSION = "4"

# third party

# relative
if TYPE_CHECKING:
    # relative
    from .syft_object import SyftObject


class SyftObjectRegistry:
    __object_transform_registry__: dict[str, Callable] = {}
    __object_serialization_registry__: dict[str, dict[int, tuple]] = {}

    @classmethod
    def register_cls(
        cls, canonical_name: str, version: int, serde_attributes: tuple
    ) -> None:
        if canonical_name not in cls.__object_serialization_registry__:
            cls.__object_serialization_registry__[canonical_name] = {}
        cls.__object_serialization_registry__[canonical_name][version] = (
            serde_attributes
        )

    @classmethod
    def get_versions(cls, canonical_name: str) -> list[int]:
        available_versions: dict = cls.__object_serialization_registry__.get(
            canonical_name,
            {},
        )
        return list(available_versions.keys())

    @classmethod
    def get_canonical_name(cls, obj: Any) -> str:
        # if is_type:
        #     # TODO: this is different for builtin types, make more generic
        #     return "ModelMetaclass"
        is_type = isinstance(obj, type)

        res = getattr(obj, "__canonical_name__", None)
        if res is not None and not is_type:
            return res
        else:
            fqn = get_fully_qualified_name(obj)
            return fqn

    @classmethod
    def get_serde_properties(cls, canonical_name: str, version: int) -> tuple:
        return cls.__object_serialization_registry__[canonical_name][version]

    @classmethod
    def get_serde_class(cls, canonical_name: str, version: int) -> type["SyftObject"]:
        serde_properties = cls.get_serde_properties(canonical_name, version)
        return serde_properties[7]

    @classmethod
    def get_serde_properties_bw_compatible(
        cls, fqn: str, canonical_name: str, version: int
    ) -> tuple:
        # relative
        from ..serde.recursive import TYPE_BANK

        if canonical_name != "" and canonical_name is not None:
            return cls.get_serde_properties(canonical_name, version)
        else:
            # this is for backward compatibility with 0.8.6
            try:
                # relative
                from ..protocol.data_protocol import get_data_protocol

                serde_props = TYPE_BANK[fqn]
                klass = serde_props[7]
                is_syftobject = hasattr(klass, "__canonical_name__")
                if is_syftobject:
                    canonical_name = klass.__canonical_name__
                    dp = get_data_protocol()
                    try:
                        version_mutations = dp.protocol_history[
                            SYFT_086_PROTOCOL_VERSION
                        ]["object_versions"][canonical_name]
                    except Exception:
                        print(f"could not find {canonical_name} in protocol history")
                        raise

                    version_086 = max(
                        [
                            int(k)
                            for k, v in version_mutations.items()
                            if v["action"] == "add"
                        ]
                    )
                    try:
                        res = cls.get_serde_properties(canonical_name, version_086)

                    except Exception:
                        print(
                            f"could not find {canonical_name} {version_086} in ObjectRegistry"
                        )
                        raise
                    return res
                else:
                    # TODO, add refactoring for non syftobject versions
                    canonical_name = fqn
                    version = 1
                    return cls.get_serde_properties(canonical_name, version)
            except Exception as e:
                print(e)
                raise

    @classmethod
    def has_serde_class(
        cls, fqn: str, canonical_name: str | None, version: int
    ) -> bool:
        # relative
        from ..serde.recursive import TYPE_BANK

        if canonical_name != "" and canonical_name is not None:
            return (
                canonical_name in cls.__object_serialization_registry__
                and version in cls.__object_serialization_registry__[canonical_name]
            )
        else:
            # this is for backward compatibility with 0.8.6
            return fqn in TYPE_BANK

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
                if issubclass(type_to_mro, SyftBaseObject):
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
