# stdlib
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import KeysView
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from collections.abc import Sequence
from collections.abc import Set
from hashlib import sha256
import inspect
from inspect import Signature
import re
import traceback
import types
from types import NoneType
from types import UnionType
import typing
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from typing import get_args
from typing import get_origin

# third party
import pandas as pd
import pydantic
from pydantic import ConfigDict
from pydantic import EmailStr
from pydantic import Field
from pydantic import model_validator
from pydantic.fields import PydanticUndefined
from result import OkErr
from typeguard import check_type
from typing_extensions import Self

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.recursive_primitives import recursive_serde_register_type
from ..serde.serialize import _serialize as serialize
from ..util.autoreload import autoreload_enabled
from ..util.markdown import as_markdown_python_code
from ..util.notebook_ui.notebook_addons import create_table_template
from ..util.util import aggressive_set_attr
from ..util.util import full_name_with_qualname
from ..util.util import get_qualname_for
from .dicttuple import DictTuple
from .syft_metaclass import Empty
from .syft_metaclass import PartialModelMetaclass
from .uid import UID

if TYPE_CHECKING:
    # relative
    from ..service.sync.diff_state import AttrDiff

IntStr = int | str
AbstractSetIntStr = Set[IntStr]
MappingIntStrAny = Mapping[IntStr, Any]


SYFT_OBJECT_VERSION_1 = 1
SYFT_OBJECT_VERSION_2 = 2
SYFT_OBJECT_VERSION_3 = 3
SYFT_OBJECT_VERSION_4 = 4

supported_object_versions = [
    SYFT_OBJECT_VERSION_1,
    SYFT_OBJECT_VERSION_2,
    SYFT_OBJECT_VERSION_3,
    SYFT_OBJECT_VERSION_4,
]

HIGHEST_SYFT_OBJECT_VERSION = max(supported_object_versions)
LOWEST_SYFT_OBJECT_VERSION = min(supported_object_versions)


# These attributes are dynamically added based on node/client
# that is interaction with the SyftObject
DYNAMIC_SYFT_ATTRIBUTES = [
    "syft_node_location",
    "syft_client_verify_key",
]


def _is_optional(x: Any) -> bool:
    return get_origin(x) in (Optional, UnionType, Union) and any(
        arg is NoneType for arg in get_args(x)
    )


def _get_optional_inner_type(x: Any) -> Any:
    if get_origin(x) not in (Optional, UnionType, Union):
        return x

    args = get_args(x)

    if not any(arg is NoneType for arg in args):
        return x

    non_none = [arg for arg in args if arg is not NoneType]
    return non_none[0] if len(non_none) == 1 else x


class SyftHashableObject:
    __hash_exclude_attrs__: list = []

    def __hash__(self) -> int:
        return int.from_bytes(self.__sha256__(), byteorder="big")

    def __sha256__(self) -> bytes:
        self.__hash_exclude_attrs__.extend(DYNAMIC_SYFT_ATTRIBUTES)
        _bytes = serialize(self, to_bytes=True, for_hashing=True)
        return sha256(_bytes).digest()

    def hash(self) -> str:
        return self.__sha256__().hex()


class SyftBaseObject(pydantic.BaseModel, SyftHashableObject):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # the name which doesn't change even when there are multiple classes
    __canonical_name__: str
    __version__: int  # data is always versioned

    syft_node_location: UID | None = Field(default=None, exclude=True)
    syft_client_verify_key: SyftVerifyKey | None = Field(default=None, exclude=True)

    def _set_obj_location_(self, node_uid: UID, credentials: SyftVerifyKey) -> None:
        self.syft_node_location = node_uid
        self.syft_client_verify_key = credentials


class Context(SyftBaseObject):
    __canonical_name__ = "Context"
    __version__ = SYFT_OBJECT_VERSION_2

    pass


class SyftObjectRegistry:
    __object_version_registry__: dict[
        str, type["SyftObject"] | type["SyftObjectRegistry"]
    ] = {}
    __object_transform_registry__: dict[str, Callable] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__canonical_name__") and hasattr(cls, "__version__"):
            mapping_string = f"{cls.__canonical_name__}_{cls.__version__}"

            if (
                mapping_string in cls.__object_version_registry__
                and not autoreload_enabled()
            ):
                current_cls = cls.__object_version_registry__[mapping_string]
                if cls == current_cls:
                    # same class so noop
                    return None

                # user code is reinitialized which means it might have a new address
                # in memory so for that we can just skip
                if "syft.user" in cls.__module__:
                    # this happens every time we reload the user code
                    return None
                else:
                    # this shouldn't happen and is usually a mistake of reusing the
                    # same __canonical_name__ and __version__ in two classes
                    raise Exception(f"Duplicate mapping for {mapping_string} and {cls}")
            else:
                # only if the cls has not been registered do we want to register it
                cls.__object_version_registry__[mapping_string] = cls

    @classmethod
    def versioned_class(
        cls, name: str, version: int
    ) -> type["SyftObject"] | type["SyftObjectRegistry"] | None:
        mapping_string = f"{name}_{version}"
        if mapping_string not in cls.__object_version_registry__:
            return None
        return cls.__object_version_registry__[mapping_string]

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
                if mapping_string in cls.__object_transform_registry__:
                    return cls.__object_transform_registry__[mapping_string]
        raise Exception(
            f"No mapping found for: {type_from} to {type_to} in"
            f"the registry: {cls.__object_transform_registry__.keys()}"
        )


class SyftMigrationRegistry:
    __migration_version_registry__: dict[str, dict[int, str]] = {}
    __migration_transform_registry__: dict[str, dict[str, Callable]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Populate the `__migration_version_registry__` dictionary with format
        __migration_version_registry__ = {
            "canonical_name": {version_number: "klass_name"}
        }
        For example
        __migration_version_registry__ = {
            'APIEndpoint': {1: 'syft.client.api.APIEndpoint'},
            'Action':      {1: 'syft.service.action.action_object.Action'},
        }
        """
        super().__init_subclass__(**kwargs)
        klass = type(cls) if not isinstance(cls, type) else cls
        cls.register_version(klass=klass)

    @classmethod
    def register_version(cls, klass: type) -> None:
        if hasattr(klass, "__canonical_name__") and hasattr(klass, "__version__"):
            mapping_string = klass.__canonical_name__
            klass_version = klass.__version__
            fqn = f"{klass.__module__}.{klass.__name__}"

            if (
                mapping_string in cls.__migration_version_registry__
                and not autoreload_enabled()
            ):
                versions = cls.__migration_version_registry__[mapping_string]
                versions[klass_version] = fqn
            else:
                # only if the cls has not been registered do we want to register it
                cls.__migration_version_registry__[mapping_string] = {
                    klass_version: fqn
                }

    @classmethod
    def get_versions(cls, canonical_name: str) -> list[int]:
        available_versions: dict = cls.__migration_version_registry__.get(
            canonical_name,
            {},
        )
        return list(available_versions.keys())

    @classmethod
    def register_transform(
        cls, klass_type_str: str, version_from: int, version_to: int, method: Callable
    ) -> None:
        """
        Populate the __migration_transform_registry__ dictionary with format
        __migration_version_registry__ = {
            "canonical_name": {"version_from x version_to": <function transform_function>}
        }
        For example
        {'NodeMetadata': {'1x2': <function transform_function>,
                          '2x1': <function transform_function>}}
        """
        if klass_type_str not in cls.__migration_version_registry__:
            raise Exception(f"{klass_type_str} is not yet registered.")

        available_versions = cls.__migration_version_registry__[klass_type_str]

        versions_exists = (
            version_from in available_versions and version_to in available_versions
        )

        if versions_exists:
            mapping_string = f"{version_from}x{version_to}"
            if klass_type_str not in cls.__migration_transform_registry__:
                cls.__migration_transform_registry__[klass_type_str] = {}
            cls.__migration_transform_registry__[klass_type_str][mapping_string] = (
                method
            )
        else:
            raise Exception(
                f"Available versions for {klass_type_str} are: {available_versions}."
                f"You're trying to add a transform from version: {version_from} to version: {version_to}"
            )

    @classmethod
    def get_migration(
        cls, type_from: type[SyftBaseObject], type_to: type[SyftBaseObject]
    ) -> Callable:
        for type_from_mro in type_from.mro():
            if (
                issubclass(type_from_mro, SyftBaseObject)
                and type_from_mro != SyftBaseObject
            ):
                klass_from = type_from_mro.__canonical_name__
                version_from = type_from_mro.__version__

                for type_to_mro in type_to.mro():
                    if (
                        issubclass(type_to_mro, SyftBaseObject)
                        and type_to_mro != SyftBaseObject
                    ):
                        klass_to = type_to_mro.__canonical_name__
                        version_to = type_to_mro.__version__

                    if klass_from == klass_to:
                        mapping_string = f"{version_from}x{version_to}"
                        if (
                            mapping_string
                            in cls.__migration_transform_registry__[klass_from]
                        ):
                            return cls.__migration_transform_registry__[klass_from][
                                mapping_string
                            ]
        raise ValueError(
            f"No migration found for class type: {type_from} to "
            f"type: {type_to} in the migration registry."
        )

    @classmethod
    def get_migration_for_version(
        cls, type_from: type[SyftBaseObject], version_to: int
    ) -> Callable:
        canonical_name = type_from.__canonical_name__
        for type_from_mro in type_from.mro():
            if (
                issubclass(type_from_mro, SyftBaseObject)
                and type_from_mro != SyftBaseObject
            ):
                klass_from = type_from_mro.__canonical_name__
                if klass_from != canonical_name:
                    continue
                version_from = type_from_mro.__version__
                mapping_string = f"{version_from}x{version_to}"
                if (
                    mapping_string
                    in cls.__migration_transform_registry__[
                        type_from.__canonical_name__
                    ]
                ):
                    return cls.__migration_transform_registry__[klass_from][
                        mapping_string
                    ]

        raise Exception(
            f"No migration found for class type: {type_from} to "
            f"version: {version_to} in the migration registry."
        )


print_type_cache: dict = defaultdict(list)


base_attrs_sync_ignore = [
    "syft_node_location",
    "syft_client_verify_key",
]


class SyftObject(SyftBaseObject, SyftObjectRegistry, SyftMigrationRegistry):
    __canonical_name__ = "SyftObject"
    __version__ = SYFT_OBJECT_VERSION_2

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={UID: str},
    )

    # all objects have a UID
    id: UID

    # # move this to transforms
    @model_validator(mode="before")
    @classmethod
    def make_id(cls, values: Any) -> Any:
        if isinstance(values, dict):
            id_field = cls.model_fields["id"]
            if "id" not in values and id_field.is_required():
                values["id"] = id_field.annotation()
        return values

    __attr_searchable__: ClassVar[
        list[str]
    ] = []  # keys which can be searched in the ORM
    __attr_unique__: ClassVar[list[str]] = []
    # the unique keys for the particular Collection the objects will be stored in
    __serde_overrides__: dict[
        str, Sequence[Callable]
    ] = {}  # List of attributes names which require a serde override.
    __owner__: str

    __repr_attrs__: ClassVar[list[str]] = []  # show these in html repr collections
    __attr_custom_repr__: ClassVar[list[str] | None] = (
        None  # show these in html repr of an object
    )
    __validate_private_attrs__: ClassVar[bool] = True

    def __syft_get_funcs__(self) -> list[tuple[str, Signature]]:
        funcs = print_type_cache[type(self)]
        if len(funcs) > 0:
            return funcs

        for attr in dir(type(self)):
            obj = getattr(type(self), attr, None)
            if (
                "SyftObject" in getattr(obj, "__qualname__", "")
                and callable(obj)
                and not isinstance(obj, type)
                and not attr.startswith("__")
            ):
                sig = inspect.signature(obj)
                funcs.append((attr, sig))

        print_type_cache[type(self)] = funcs
        return funcs

    def __repr__(self) -> str:
        try:
            fqn = full_name_with_qualname(type(self))
            return fqn
        except Exception:
            return str(type(self))

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_debug_(self) -> str:
        class_name = get_qualname_for(type(self))
        _repr_str = f"class {class_name}:\n"
        fields = getattr(self, "model_fields", {})
        for attr in fields.keys():
            if attr in DYNAMIC_SYFT_ATTRIBUTES:
                continue
            value = getattr(self, attr, "<Missing>")
            value_type = full_name_with_qualname(type(attr))
            value_type = value_type.replace("builtins.", "")
            if hasattr(value, "syft_action_data_str_"):
                value = value.syft_action_data_str_
            value = f'"{value}"' if isinstance(value, str) else value
            _repr_str += f"  {attr}: {value_type} = {value}\n"
        return _repr_str

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        s_indent = " " * indent * 2
        class_name = get_qualname_for(type(self))
        if self.__attr_custom_repr__ is not None:
            fields = self.__attr_custom_repr__
        elif self.__repr_attrs__ is not None:
            fields = self.__repr_attrs__
        else:
            fields = list(getattr(self, "__fields__", {}).keys())  # type: ignore[unreachable]

        if "id" not in fields:
            fields = ["id"] + fields

        dynam_attrs = set(DYNAMIC_SYFT_ATTRIBUTES)
        fields = [x for x in fields if x not in dynam_attrs]
        _repr_str = f"{s_indent}class {class_name}:\n"
        for attr in fields:
            value: Any = self
            # if it's a compound string
            if "." in attr:
                # break it into it's bits & fetch the attr
                for _attr in attr.split("."):
                    value = getattr(value, _attr, "<Missing>")
            else:
                value = getattr(value, attr, "<Missing>")

            value_type = full_name_with_qualname(type(attr))
            value_type = value_type.replace("builtins.", "")
            # If the object has a special representation when nested we will use that instead
            if hasattr(value, "__repr_syft_nested__"):
                value = value.__repr_syft_nested__()
            if isinstance(value, list):
                value = [
                    (
                        elem.__repr_syft_nested__()
                        if hasattr(elem, "__repr_syft_nested__")
                        else elem
                    )
                    for elem in value
                ]
            value = f'"{value}"' if isinstance(value, str) else value
            _repr_str += f"{s_indent}  {attr}: {value_type} = {value}\n"

        # _repr_str += "\n"
        # fqn = full_name_with_qualname(type(self))
        # _repr_str += f'fqn = "{fqn}"\n'
        # _repr_str += f"mro = {[t.__name__ for t in type(self).mro()]}"

        # _repr_str += "\n\ncallables = [\n"
        # for func, sig in self.__syft_get_funcs__():
        #     _repr_str += f"  {func}{sig}: pass\n"
        # _repr_str += f"]"
        # return _repr_str
        if wrap_as_python:
            return as_markdown_python_code(_repr_str)
        else:
            return _repr_str

    # allows splatting with **
    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    # allows splatting with **
    def __getitem__(self, key: str | int) -> Any:
        return self.__dict__.__getitem__(key)  # type: ignore

    def _upgrade_version(self, latest: bool = True) -> "SyftObject":
        constructor = SyftObjectRegistry.versioned_class(
            name=self.__canonical_name__, version=self.__version__ + 1
        )
        if not constructor:
            return self
        else:
            # should we do some kind of recursive upgrades?
            upgraded = constructor._from_previous_version(self)
            if latest:
                upgraded = upgraded._upgrade_version(latest=latest)
            return upgraded

    # transform from one supported type to another
    def to(self, projection: type, context: Context | None = None) -> Any:
        # 🟡 TODO 19: Could we do an mro style inheritence conversion? Risky?
        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self, context)

    def to_dict(
        self, exclude_none: bool = False, exclude_empty: bool = False
    ) -> dict[str, Any]:
        new_dict = {}
        for k, v in dict(self).items():
            # exclude dynamically added syft attributes
            if k in DYNAMIC_SYFT_ATTRIBUTES:
                continue
            if exclude_empty and v is Empty:
                continue
            if exclude_none and v is None:
                continue
            new_dict[k] = v
        return new_dict

    def __post_init__(self) -> None:
        pass

    def _syft_set_validate_private_attrs_(self, **kwargs: Any) -> None:
        if not self.__validate_private_attrs__:
            return
        # Validate and set private attributes
        # https://github.com/pydantic/pydantic/issues/2105
        annotations = typing.get_type_hints(self.__class__)
        for attr, decl in self.__private_attributes__.items():
            value = kwargs.get(attr, decl.get_default())
            var_annotation = annotations.get(attr)
            if value is not PydanticUndefined:
                if var_annotation is not None:
                    # Otherwise validate value against the variable annotation
                    check_type(value, var_annotation)
                setattr(self, attr, value)
            else:
                if not _is_optional(var_annotation):
                    raise ValueError(
                        f"{attr}\n field required (type=value_error.missing)"
                    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._syft_set_validate_private_attrs_(**kwargs)
        self.__post_init__()

    # TODO: Check why Pydantic is removing the __hash__ method during inheritance
    def __hash__(self) -> int:
        return int.from_bytes(self.__sha256__(), byteorder="big")

    @classmethod
    def _syft_keys_types_dict(cls, attr_name: str) -> dict[str, type]:
        kt_dict = {}
        for key in getattr(cls, attr_name, []):
            if key in cls.model_fields:
                type_ = _get_optional_inner_type(cls.model_fields[key].annotation)
            else:
                try:
                    method = getattr(cls, key)
                    if isinstance(method, types.FunctionType):
                        type_ = method.__annotations__["return"]
                except Exception as e:
                    print(
                        f"Failed to get attribute from key {key} type for {cls} storage. {e}"
                    )
                    raise e
            # EmailStr seems to be lost every time the value is set even with a validator
            # this means the incoming type is str so our validators fail

            if type_ is EmailStr:
                type_ = str

            kt_dict[key] = type_
        return kt_dict

    @classmethod
    def _syft_unique_keys_dict(cls) -> dict[str, type]:
        return cls._syft_keys_types_dict("__attr_unique__")

    @classmethod
    def _syft_searchable_keys_dict(cls) -> dict[str, type]:
        return cls._syft_keys_types_dict("__attr_searchable__")

    def migrate_to(self, version: int, context: Context | None = None) -> Any:
        if self.__version__ != version:
            migration_transform = SyftMigrationRegistry.get_migration_for_version(
                type_from=type(self), version_to=version
            )
            return migration_transform(
                self,
                context,
            )
        return self

    def syft_eq(self, ext_obj: Self | None) -> bool:
        if ext_obj is None:
            return False
        attrs_to_check = self.__dict__.keys()

        obj_exclude_attrs = getattr(self, "__exclude_sync_diff_attrs__", [])
        for attr in attrs_to_check:
            if attr not in base_attrs_sync_ignore and attr not in obj_exclude_attrs:
                obj_attr = getattr(self, attr)
                ext_obj_attr = getattr(ext_obj, attr)
                if hasattr(obj_attr, "syft_eq") and not inspect.isclass(obj_attr):
                    if not obj_attr.syft_eq(ext_obj=ext_obj_attr):
                        return False
                elif obj_attr != ext_obj_attr:
                    return False
        return True

    def syft_get_diffs(self, ext_obj: Self) -> list["AttrDiff"]:
        # self is low, ext is high
        # relative
        from ..service.sync.diff_state import AttrDiff
        from ..service.sync.diff_state import ListDiff

        diff_attrs = []

        # Sanity check
        if self.id != ext_obj.id:
            raise Exception("Not the same id for low side and high side requests")

        attrs_to_check = self.__dict__.keys()

        obj_exclude_attrs = getattr(self, "__exclude_sync_diff_attrs__", [])
        for attr in attrs_to_check:
            if attr not in base_attrs_sync_ignore and attr not in obj_exclude_attrs:
                obj_attr = getattr(self, attr)
                ext_obj_attr = getattr(ext_obj, attr)

                if isinstance(obj_attr, list) and isinstance(ext_obj_attr, list):
                    list_diff = ListDiff.from_lists(
                        attr_name=attr, low_list=obj_attr, high_list=ext_obj_attr
                    )
                    if not list_diff.is_empty:
                        diff_attrs.append(list_diff)

                # TODO: to the same check as above for Dicts when we use them
                else:
                    cmp = obj_attr.__eq__
                    if hasattr(obj_attr, "syft_eq"):
                        cmp = obj_attr.syft_eq

                    if not cmp(ext_obj_attr):
                        diff_attr = AttrDiff(
                            attr_name=attr,
                            low_attr=obj_attr,
                            high_attr=ext_obj_attr,
                        )
                        diff_attrs.append(diff_attr)
        return diff_attrs

    ## OVERRIDING pydantic.BaseModel.__getattr__
    ## return super().__getattribute__(item) -> return self.__getattribute__(item)
    ## so that ActionObject.__getattribute__ works properly,
    ## raising AttributeError when underlying object does not have the attribute
    if not typing.TYPE_CHECKING:
        # We put `__getattr__` in a non-TYPE_CHECKING block because otherwise, mypy allows arbitrary attribute access

        def __getattr__(self, item: str) -> Any:
            private_attributes = object.__getattribute__(self, "__private_attributes__")
            if item in private_attributes:
                attribute = private_attributes[item]
                if hasattr(attribute, "__get__"):
                    return attribute.__get__(self, type(self))  # type: ignore

                try:
                    # Note: self.__pydantic_private__ cannot be None if self.__private_attributes__ has items
                    return self.__pydantic_private__[item]  # type: ignore
                except KeyError as exc:
                    raise AttributeError(
                        f"{type(self).__name__!r} object has no attribute {item!r}"
                    ) from exc
            else:
                # `__pydantic_extra__` can fail to be set if the model is not yet fully initialized.
                # See `BaseModel.__repr_args__` for more details
                try:
                    pydantic_extra = object.__getattribute__(self, "__pydantic_extra__")
                except AttributeError:
                    pydantic_extra = None

                if pydantic_extra is not None:
                    try:
                        return pydantic_extra[item]
                    except KeyError as exc:
                        raise AttributeError(
                            f"{type(self).__name__!r} object has no attribute {item!r}"
                        ) from exc
                else:
                    if hasattr(self.__class__, item):
                        return self.__getattribute__(
                            item
                        )  # Raises AttributeError if appropriate
                    else:
                        # this is the current error
                        raise AttributeError(
                            f"{type(self).__name__!r} object has no attribute {item!r}"
                        )


def short_qual_name(name: str) -> str:
    # If the name is a qualname of formax a.b.c.d we will only get d
    # otherwise this will leave it like it is
    return name.split(".")[-1]


def short_uid(uid: UID | None) -> str | None:
    if uid is None:
        return uid
    else:
        return str(uid)[:6] + "..."


def get_repr_values_table(
    _self: Mapping | Iterable,
    is_homogenous: bool,
    extra_fields: list | None = None,
) -> dict:
    if extra_fields is None:
        extra_fields = []

    cols = defaultdict(list)
    for item in iter(_self.items() if isinstance(_self, Mapping) else _self):
        # unpack dict
        if isinstance(_self, Mapping):
            key, item = item
            cols["key"].append(key)

        # get id
        id_ = getattr(item, "id", None)
        include_id = getattr(item, "__syft_include_id_coll_repr__", True)
        if id_ is not None and include_id:
            cols["id"].append({"value": str(id_), "type": "clipboard"})

        if type(item) == type:
            t = full_name_with_qualname(item)
        else:
            try:
                t = item.__class__.__name__
            except Exception:
                t = item.__repr__()

        if not is_homogenous:
            cols["type"].append(t)

        # if has _coll_repr_

        if hasattr(item, "_coll_repr_"):
            ret_val = item._coll_repr_()
            if "id" in ret_val:
                del ret_val["id"]
            for key in ret_val.keys():
                cols[key].append(ret_val[key])
        else:
            for field in extra_fields:
                value = item
                try:
                    attrs = field.split(".")
                    for i, attr in enumerate(attrs):
                        # find indexing like abc[1]
                        res = re.search(r"\[[+-]?\d+\]", attr)
                        has_index = False
                        if res:
                            has_index = True
                            index_str = res.group()
                            index = int(index_str.replace("[", "").replace("]", ""))
                            attr = attr.replace(index_str, "")

                        value = getattr(value, attr, None)
                        if isinstance(value, list) and has_index:
                            value = value[index]
                        # If the object has a special representation when nested we will use that instead
                        if (
                            hasattr(value, "__repr_syft_nested__")
                            and i == len(attrs) - 1
                        ):
                            value = value.__repr_syft_nested__()
                        if (
                            isinstance(value, list)
                            and i == len(attrs) - 1
                            and len(value) > 0
                            and hasattr(value[0], "__repr_syft_nested__")
                        ):
                            value = [
                                (
                                    x.__repr_syft_nested__()
                                    if hasattr(x, "__repr_syft_nested__")
                                    else x
                                )
                                for x in value
                            ]
                    if value is None:
                        value = "n/a"

                except Exception as e:
                    print(e)
                    value = None
                cols[field].append(str(value))

    df = pd.DataFrame(cols)

    if "created_at" in df.columns:
        df.sort_values(by="created_at", ascending=False, inplace=True)

    return df.to_dict("records")  # type: ignore


def list_dict_repr_html(self: Mapping | Set | Iterable) -> str:
    try:
        max_check = 1
        items_checked = 0
        has_syft = False
        extra_fields: list = []
        if isinstance(self, Mapping):
            values: Any = list(self.values())
        elif isinstance(self, Set):
            values = list(self)
        else:
            values = self

        if len(values) == 0:
            return self.__repr__()

        for item in iter(self.values() if isinstance(self, Mapping) else self):
            items_checked += 1
            if items_checked > max_check:
                break

            if hasattr(type(item), "mro") and type(item) != type:
                mro: list | str = type(item).mro()
            elif hasattr(item, "mro") and type(item) != type:
                mro = item.mro()
            else:
                mro = str(self)

            if "syft" in str(mro).lower():
                has_syft = True
                extra_fields = getattr(item, "__repr_attrs__", [])
                break

        if has_syft:
            # if custom_repr:
            table_icon = None
            if hasattr(values[0], "icon"):
                table_icon = values[0].icon
            # this is a list of dicts
            is_homogenous = len({type(x) for x in values}) == 1
            # third party
            first_value = values[0]
            if is_homogenous:
                cls_name = first_value.__class__.__name__
            else:
                cls_name = ""
            try:
                vals = get_repr_values_table(
                    self, is_homogenous, extra_fields=extra_fields
                )
            except Exception:
                return str(self)

            return create_table_template(
                vals,
                f"{cls_name} {self.__class__.__name__.capitalize()}",
                table_icon=table_icon,
            )

    except Exception as e:
        print(
            f"error representing {type(self)} of objects. {e}, {traceback.format_exc()}"
        )
        pass

    # stdlib
    import html

    return html.escape(self.__repr__())


# give lists and dicts a _repr_html_ if they contain SyftObject's
aggressive_set_attr(type([]), "_repr_html_", list_dict_repr_html)
aggressive_set_attr(type({}), "_repr_html_", list_dict_repr_html)
aggressive_set_attr(type(set()), "_repr_html_", list_dict_repr_html)
aggressive_set_attr(tuple, "_repr_html_", list_dict_repr_html)


class StorableObjectType:
    def to(self, projection: type, context: Context | None = None) -> Any:
        # 🟡 TODO 19: Could we do an mro style inheritence conversion? Risky?
        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self, context)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


TupleGenerator = Generator[tuple[str, Any], None, None]


class PartialSyftObject(SyftObject, metaclass=PartialModelMetaclass):
    """Syft Object to which partial arguments can be provided."""

    __canonical_name__ = "PartialSyftObject"
    __version__ = SYFT_OBJECT_VERSION_2

    def __iter__(self) -> TupleGenerator:
        yield from ((k, v) for k, v in super().__iter__() if v is not Empty)


recursive_serde_register_type(PartialSyftObject)


def attach_attribute_to_syft_object(result: Any, attr_dict: dict[str, Any]) -> Any:
    constructor = None
    extra_args = []

    single_entity = False

    if isinstance(result, OkErr):
        constructor = type(result)
        result = result.value

    if isinstance(result, MutableMapping):
        iterable_keys: Iterable = result.keys()
    elif isinstance(result, MutableSequence):
        iterable_keys = range(len(result))
    elif isinstance(result, tuple):
        iterable_keys = range(len(result))
        constructor = type(result)
        if isinstance(result, DictTuple):
            extra_args.append(result.keys())
        result = list(result)
    else:
        iterable_keys = range(1)
        result = [result]
        single_entity = True

    for key in iterable_keys:
        _object = result[key]
        # if object is SyftBaseObject,
        # then attach the value to the attribute
        # on the object
        if isinstance(_object, SyftBaseObject):
            for attr_name, attr_value in attr_dict.items():
                setattr(_object, attr_name, attr_value)

            for field_name, attr in _object.__dict__.items():
                updated_attr = attach_attribute_to_syft_object(attr, attr_dict)
                setattr(_object, field_name, updated_attr)
        result[key] = _object

    wrapped_result = result[0] if single_entity else result
    if constructor is not None:
        wrapped_result = constructor(wrapped_result, *extra_args)

    return wrapped_result
