# stdlib
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import KeysView
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Set
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import cache
from functools import total_ordering
from hashlib import sha256
import inspect
from inspect import Signature
import logging
import types
from types import NoneType
from types import UnionType
import typing
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

# third party
import pydantic
from pydantic import ConfigDict
from pydantic import EmailStr
from pydantic import Field
from pydantic import model_validator
from pydantic.fields import PydanticUndefined
from typeguard import check_type
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..serde.serialize import _serialize as serialize
from ..server.credentials import SyftVerifyKey
from ..util.autoreload import autoreload_enabled
from ..util.markdown import as_markdown_python_code
from ..util.notebook_ui.components.tabulator_template import build_tabulator_table
from ..util.util import aggressive_set_attr
from ..util.util import full_name_with_qualname
from ..util.util import get_qualname_for
from .result import Err
from .result import Ok
from .syft_equals import _syft_equals
from .syft_metaclass import Empty
from .syft_metaclass import PartialModelMetaclass
from .syft_object_registry import SyftObjectRegistry
from .uid import UID

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # relative
    from ..client.api import SyftAPI
    from ..service.sync.diff_state import AttrDiff

IntStr = int | str
AbstractSetIntStr = Set[IntStr]
MappingIntStrAny = Mapping[IntStr, Any]
T = TypeVar("T")


SYFT_OBJECT_VERSION_1 = 1
SYFT_OBJECT_VERSION_2 = 2
SYFT_OBJECT_VERSION_3 = 3
SYFT_OBJECT_VERSION_4 = 4
SYFT_OBJECT_VERSION_5 = 5
SYFT_OBJECT_VERSION_6 = 6

supported_object_versions = [
    SYFT_OBJECT_VERSION_1,
    SYFT_OBJECT_VERSION_2,
    SYFT_OBJECT_VERSION_3,
    SYFT_OBJECT_VERSION_4,
    SYFT_OBJECT_VERSION_5,
    SYFT_OBJECT_VERSION_6,
]

HIGHEST_SYFT_OBJECT_VERSION = max(supported_object_versions)
LOWEST_SYFT_OBJECT_VERSION = min(supported_object_versions)


# These attributes are dynamically added based on server/client
# that is interaction with the SyftObject
DYNAMIC_SYFT_ATTRIBUTES = [
    "syft_server_location",
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

    syft_server_location: UID | None = Field(default=None, exclude=True)
    syft_client_verify_key: SyftVerifyKey | None = Field(default=None, exclude=True)

    def _set_obj_location_(self, server_uid: UID, credentials: SyftVerifyKey) -> None:
        self.syft_server_location = server_uid
        self.syft_client_verify_key = credentials

    def get_api(
        self,
        server_uid: UID | None = None,
        user_verify_key: SyftVerifyKey | None = None,
    ) -> "SyftAPI":
        if server_uid is None:
            server_uid = self.syft_server_location

        if user_verify_key is None:
            user_verify_key = self.syft_client_verify_key

        # relative
        from ..client.api import APIRegistry

        return APIRegistry.api_for(
            server_uid=server_uid,
            user_verify_key=user_verify_key,
        ).unwrap(
            public_message=f"Can't access Syft API using this object. You must login to {self.syft_server_location}"
        )

    def get_api_wrapped(self):  # type: ignore
        # relative
        from ..client.api import APIRegistry

        return APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )


class Context(SyftBaseObject):
    __canonical_name__ = "Context"
    __version__ = SYFT_OBJECT_VERSION_1

    pass


@cache
def cached_get_type_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


class SyftMigrationRegistry:
    __migration_version_registry__: dict[str, dict[int, str]] = {}
    __migration_function_registry__: dict[str, dict[str, Callable]] = {}

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

    # @classmethod
    # def get_versions(cls, canonical_name: str) -> list[int]:
    #     available_versions: dict = cls.__migration_version_registry__.get(
    #         canonical_name,
    #         {},
    #     )
    #     return list(available_versions.keys())

    @classmethod
    def register_migration_function(
        cls, klass_type_str: str, version_from: int, version_to: int, method: Callable
    ) -> None:
        """
        Populate the __migration_transform_registry__ dictionary with format
        __migration_version_registry__ = {
            "canonical_name": {"version_from x version_to": <function transform_function>}
        }
        For example
        {'ServerMetadata': {'1x2': <function transform_function>,
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
            if klass_type_str not in cls.__migration_function_registry__:
                cls.__migration_function_registry__[klass_type_str] = {}
            cls.__migration_function_registry__[klass_type_str][mapping_string] = method
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
                            in cls.__migration_function_registry__[klass_from]
                        ):
                            return cls.__migration_function_registry__[klass_from][
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
                    in cls.__migration_function_registry__[type_from.__canonical_name__]
                ):
                    return cls.__migration_function_registry__[klass_from][
                        mapping_string
                    ]

        raise Exception(
            f"No migration found for class type: {type_from} to "
            f"version: {version_to} in the migration registry."
        )


print_type_cache: dict = defaultdict(list)


base_attrs_sync_ignore = [
    "syft_server_location",
    "syft_client_verify_key",
]


@serializable()
class SyftObjectVersioned(SyftBaseObject, SyftMigrationRegistry):
    __canonical_name__ = "SyftObjectVersioned"
    __version__ = SYFT_OBJECT_VERSION_1


@serializable()
@total_ordering
class BaseDateTime(SyftObjectVersioned):
    __canonical_name__ = "BaseDateTime"
    __version__ = SYFT_OBJECT_VERSION_1
    # id: UID | None = None  # type: ignore
    utc_timestamp: float

    @classmethod
    def now(cls) -> Self:
        return cls(utc_timestamp=datetime.now(timezone.utc).timestamp())

    def __str__(self) -> str:
        utc_datetime = datetime.fromtimestamp(self.utc_timestamp, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S")

    def __hash__(self) -> int:
        return hash(self.utc_timestamp)

    def __sub__(self, other: Self) -> timedelta:
        res = timedelta(seconds=self.utc_timestamp - other.utc_timestamp)
        return res

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return False
        return self.utc_timestamp == other.utc_timestamp

    def __lt__(self, other: Self) -> bool:
        return self.utc_timestamp < other.utc_timestamp


EXCLUDED_FROM_SIGNATURE = set(
    DYNAMIC_SYFT_ATTRIBUTES + ["created_date", "updated_date", "deleted_date"]
)


@serializable()
class SyftObject(SyftObjectVersioned):
    __canonical_name__ = "SyftObject"
    __version__ = SYFT_OBJECT_VERSION_1

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={UID: str},
    )

    # all objects have a UID
    id: UID
    created_date: BaseDateTime | None = None
    updated_date: BaseDateTime | None = None
    deleted_date: BaseDateTime | None = None

    # # move this to transforms
    @model_validator(mode="before")
    @classmethod
    def make_id(cls, values: Any) -> Any:
        if isinstance(values, dict):
            id_field = cls.model_fields["id"]
            if "id" not in values and id_field.is_required():
                values["id"] = id_field.annotation()
        return values

    __order_by__: ClassVar[tuple[str, str]] = ("_created_at", "asc")
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
    __table_coll_widths__: ClassVar[list[str] | None] = None
    __table_sort_attr__: ClassVar[str | None] = None

    def refresh(self) -> None:
        try:
            api = self._get_api()
            new_object = api.services.migration._get_object(
                uid=self.id, object_type=type(self)
            )
            if type(new_object) == type(self):
                self.__dict__.update(new_object.__dict__)
        except Exception as _:
            return

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

    # transform from one supported type to another
    def to(self, projection: type[T], context: Context | None = None) -> T:
        # relative

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
        annotations = cached_get_type_hints(self.__class__)
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
                    logger.error(
                        f"Failed to get attribute from key {key} type for {cls} storage.",
                        exc_info=e,
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
        obj_exclude_attrs.extend(["created_date", "updated_date", "deleted_date"])
        for attr in attrs_to_check:
            if attr not in base_attrs_sync_ignore and attr not in obj_exclude_attrs:
                obj_attr = getattr(self, attr)
                ext_obj_attr = getattr(ext_obj, attr)
                if hasattr(obj_attr, "syft_eq") and not inspect.isclass(obj_attr):
                    if not obj_attr.syft_eq(ext_obj=ext_obj_attr):
                        return False
                elif not _syft_equals(obj_attr, ext_obj_attr):
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
        obj_exclude_attrs.extend(["created_date", "updated_date", "deleted_date"])
        for attr in attrs_to_check:
            if attr not in base_attrs_sync_ignore and attr not in obj_exclude_attrs:
                obj_attr = getattr(self, attr)
                ext_obj_attr = getattr(ext_obj, attr)

                # TODO move to _syft_equals
                if isinstance(obj_attr, list) and isinstance(ext_obj_attr, list):
                    list_diff = ListDiff.from_lists(
                        attr_name=attr, low_list=obj_attr, high_list=ext_obj_attr
                    )
                    if not list_diff.is_empty:
                        diff_attrs.append(list_diff)

                else:
                    if hasattr(obj_attr, "syft_eq"):
                        is_equal = obj_attr.syft_eq(ext_obj_attr)
                    else:
                        is_equal = _syft_equals(obj_attr, ext_obj_attr)

                    if not is_equal:
                        diff_attr = AttrDiff(
                            attr_name=attr,
                            low_attr=obj_attr,
                            high_attr=ext_obj_attr,
                        )
                        diff_attrs.append(diff_attr)
        return diff_attrs

    # TODO: Move this away from here
    def _get_api(self) -> "SyftAPI":
        # relative
        from ..client.api import APIRegistry

        return APIRegistry.api_for(
            self.syft_server_location, self.syft_client_verify_key
        ).unwrap()

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


# give lists and dicts a _repr_html_ if they contain SyftObject's
aggressive_set_attr(type([]), "_repr_html_", build_tabulator_table)
aggressive_set_attr(type({}), "_repr_html_", build_tabulator_table)
aggressive_set_attr(type(set()), "_repr_html_", build_tabulator_table)
aggressive_set_attr(tuple, "_repr_html_", build_tabulator_table)


class StorableObjectType:
    def to(self, projection: type, context: Context | None = None) -> Any:
        # 🟡 TODO 19: Could we do an mro style inheritence conversion? Risky?
        # relative

        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self, context)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


TupleGenerator = Generator[tuple[str, Any], None, None]


@serializable()
class PartialSyftObject(SyftObject, metaclass=PartialModelMetaclass):
    """Syft Object to which partial arguments can be provided."""

    __canonical_name__ = "PartialSyftObject"
    __version__ = SYFT_OBJECT_VERSION_1

    def __iter__(self) -> TupleGenerator:
        yield from ((k, v) for k, v in super().__iter__() if v is not Empty)

    def apply(self, to: SyftObject) -> None:
        for k, v in self:
            setattr(to, k, v)


def attach_attribute_to_syft_object(result: Any, attr_dict: dict[str, Any]) -> None:
    iterator: Iterable

    if isinstance(result, Ok):
        iterator = (result.ok(),)
    elif isinstance(result, Err):
        iterator = (result.err(),)
    elif isinstance(result, Mapping):
        iterator = result.values()
    elif isinstance(result, Sequence):
        iterator = result
    else:
        iterator = (result,)

    for _object in iterator:
        # if object is SyftBaseObject,
        # then attach the value to the attribute
        # on the object
        if isinstance(_object, SyftBaseObject):
            for attr_name, attr_value in attr_dict.items():
                setattr(_object, attr_name, attr_value)

            for field in _object.model_fields.keys():
                attach_attribute_to_syft_object(getattr(_object, field), attr_dict)
