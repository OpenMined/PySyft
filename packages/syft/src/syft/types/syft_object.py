# stdlib
from collections import defaultdict
from collections.abc import Set
from hashlib import sha256
import inspect
from inspect import Signature
import re
import types
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import KeysView
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
import warnings

# third party
import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import EmailStr
from pydantic.fields import Undefined
from result import OkErr
from typeguard import check_type

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
from .syft_metaclass import Empty
from .syft_metaclass import PartialModelMetaclass
from .uid import UID

IntStr = Union[int, str]
AbstractSetIntStr = Set[IntStr]
MappingIntStrAny = Mapping[IntStr, Any]


SYFT_OBJECT_VERSION_1 = 1
SYFT_OBJECT_VERSION_2 = 2

supported_object_versions = [SYFT_OBJECT_VERSION_1, SYFT_OBJECT_VERSION_2]

HIGHEST_SYFT_OBJECT_VERSION = max(supported_object_versions)
LOWEST_SYFT_OBJECT_VERSION = min(supported_object_versions)


# These attributes are dynamically added based on node/client
# that is interaction with the SyftObject
DYNAMIC_SYFT_ATTRIBUTES = [
    "syft_node_location",
    "syft_client_verify_key",
]


class SyftHashableObject:
    __hash_exclude_attrs__ = []

    def __hash__(self) -> int:
        return int.from_bytes(self.__sha256__(), byteorder="big")

    def __sha256__(self) -> bytes:
        self.__hash_exclude_attrs__.extend(DYNAMIC_SYFT_ATTRIBUTES)
        _bytes = serialize(self, to_bytes=True, for_hashing=True)
        return sha256(_bytes).digest()

    def hash(self) -> str:
        return self.__sha256__().hex()


class SyftBaseObject(BaseModel, SyftHashableObject):
    class Config:
        arbitrary_types_allowed = True

    __canonical_name__: str  # the name which doesn't change even when there are multiple classes
    __version__: int  # data is always versioned

    syft_node_location: Optional[UID]
    syft_client_verify_key: Optional[SyftVerifyKey]


class Context(SyftBaseObject):
    pass


class SyftObjectRegistry:
    __object_version_registry__: Dict[str, Type["SyftObject"]] = {}
    __object_transform_registry__: Dict[str, Callable] = {}

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
    def versioned_class(cls, name: str, version: int) -> Optional[Type["SyftObject"]]:
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
        cls, type_from: Type["SyftObject"], type_to: Type["SyftObject"]
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


print_type_cache = defaultdict(list)


class SyftObject(SyftBaseObject, SyftObjectRegistry):
    __canonical_name__ = "SyftObject"
    __version__ = SYFT_OBJECT_VERSION_1

    class Config:
        arbitrary_types_allowed = True

    # all objects have a UID
    id: UID

    # # move this to transforms
    @pydantic.root_validator(pre=True)
    def make_id(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        id_field = cls.__fields__["id"]
        if "id" not in values and id_field.required:
            values["id"] = id_field.type_()
        return values

    __attr_searchable__: ClassVar[
        List[str]
    ] = []  # keys which can be searched in the ORM
    __attr_unique__: ClassVar[List[str]] = []
    # the unique keys for the particular Collection the objects will be stored in
    __serde_overrides__: Dict[
        str, Sequence[Callable]
    ] = {}  # List of attributes names which require a serde override.
    __owner__: str

    __repr_attrs__: ClassVar[List[str]] = []  # show these in html repr collections
    __attr_custom_repr__: ClassVar[
        List[str]
    ] = None  # show these in html repr of an object

    def __syft_get_funcs__(self) -> List[Tuple[str, Signature]]:
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
        fields = getattr(self, "__fields__", {})
        for attr in fields.keys():
            if attr in DYNAMIC_SYFT_ATTRIBUTES:
                continue
            value = getattr(self, attr, "<Missing>")
            value_type = full_name_with_qualname(type(attr))
            value_type = value_type.replace("builtins.", "")
            value = f'"{value}"' if isinstance(value, str) else value
            _repr_str += f"  {attr}: {value_type} = {value}\n"
        return _repr_str

    def _repr_markdown_(self, wrap_as_python=True, indent=0) -> str:
        s_indent = " " * indent * 2
        class_name = get_qualname_for(type(self))
        if self.__attr_custom_repr__ is not None:
            fields = self.__attr_custom_repr__
        elif self.__repr_attrs__ is not None:
            fields = self.__repr_attrs__
        else:
            fields = list(getattr(self, "__fields__", {}).keys())

        if "id" not in fields:
            fields = ["id"] + fields

        dynam_attrs = set(DYNAMIC_SYFT_ATTRIBUTES)
        fields = [x for x in fields if x not in dynam_attrs]
        _repr_str = f"{s_indent}class {class_name}:\n"
        for attr in fields:
            value = self
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
                    elem.__repr_syft_nested__()
                    if hasattr(elem, "__repr_syft_nested__")
                    else elem
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
    def __getitem__(self, key: str) -> Any:
        return self.__dict__.__getitem__(key)

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
    def to(self, projection: type, context: Optional[Context] = None) -> Any:
        # 🟡 TODO 19: Could we do an mro style inheritence conversion? Risky?
        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self, context)

    def to_dict(
        self, exclude_none: bool = False, exclude_empty: bool = False
    ) -> Dict[str, Any]:
        warnings.warn(
            "`SyftObject.to_dict` is deprecated and will be removed in a future version",
            PendingDeprecationWarning,
        )
        # 🟡 TODO 18: Remove to_dict and replace usage with transforms etc
        if not exclude_none and not exclude_empty:
            return self.dict()
        else:
            new_dict = {}
            for k, v in dict(self).items():
                # exclude dynamically added syft attributes
                if k in DYNAMIC_SYFT_ATTRIBUTES:
                    continue
                if exclude_empty and v is not Empty:
                    new_dict[k] = v
                if exclude_none and v is not None:
                    new_dict[k] = v
            return new_dict

    def dict(
        self,
        *,
        include: Optional[Union["AbstractSetIntStr", "MappingIntStrAny"]] = None,
        exclude: Optional[Union["AbstractSetIntStr", "MappingIntStrAny"]] = None,
        by_alias: bool = False,
        skip_defaults: Optional[bool] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ):
        if exclude is None:
            exclude = set()

        for attr in DYNAMIC_SYFT_ATTRIBUTES:
            exclude.add(attr)
        return super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

    def __post_init__(self) -> None:
        pass

    def _syft_set_validate_private_attrs_(self, **kwargs):
        # Validate and set private attributes
        # https://github.com/pydantic/pydantic/issues/2105
        for attr, decl in self.__private_attributes__.items():
            value = kwargs.get(attr, decl.get_default())
            var_annotation = self.__annotations__.get(attr)
            if value is not Undefined:
                if decl.default_factory:
                    # If the value is defined via PrivateAttr with default factory
                    value = decl.default_factory(value)
                elif var_annotation is not None:
                    # Otherwise validate value against the variable annotation
                    check_type(attr, value, var_annotation)
                setattr(self, attr, value)
            else:
                # check if the private is optional
                is_optional_attr = type(None) in getattr(var_annotation, "__args__", [])
                if not is_optional_attr:
                    raise ValueError(
                        f"{attr}\n field required (type=value_error.missing)"
                    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._syft_set_validate_private_attrs_(**kwargs)
        self.__post_init__()

    # TODO: Check why Pydantic is removing the __hash__ method during inheritance
    def __hash__(self) -> int:
        return int.from_bytes(self.__sha256__(), byteorder="big")

    @classmethod
    def _syft_keys_types_dict(cls, attr_name: str) -> Dict[str, type]:
        kt_dict = {}
        for key in getattr(cls, attr_name, []):
            if key in cls.__fields__:
                type_ = cls.__fields__[key].type_
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

            if type(type_) is type and issubclass(type_, EmailStr):
                type_ = str
            kt_dict[key] = type_
        return kt_dict

    @classmethod
    def _syft_unique_keys_dict(cls) -> Dict[str, type]:
        return cls._syft_keys_types_dict("__attr_unique__")

    @classmethod
    def _syft_searchable_keys_dict(cls) -> Dict[str, type]:
        return cls._syft_keys_types_dict("__attr_searchable__")


def short_qual_name(name: str) -> str:
    # If the name is a qualname of formax a.b.c.d we will only get d
    # otherwise this will leave it like it is
    return name.split(".")[-1]


def short_uid(uid: UID) -> str:
    if uid is None:
        return uid
    else:
        return str(uid)[:6] + "..."


def list_dict_repr_html(self) -> str:
    try:
        max_check = 1
        items_checked = 0
        has_syft = False
        extra_fields = []
        if isinstance(self, dict):
            values = list(self.values())
        else:
            values = self

        if len(values) == 0:
            return self.__repr__()

        is_homogenous = len(set([type(x) for x in values])) == 1
        for item in iter(self):
            items_checked += 1
            if items_checked > max_check:
                break
            if isinstance(self, dict):
                item = self.__getitem__(item)

            if hasattr(type(item), "mro") and type(item) != type:
                mro = type(item).mro()
            elif hasattr(item, "mro") and type(item) != type:
                mro = item.mro()
            else:
                mro = str(self)

            if "syft" in str(mro).lower():
                has_syft = True
                extra_fields = getattr(item, "__repr_attrs__", [])
                break
        if has_syft:
            # third party
            first_value = values[0]
            if is_homogenous:
                cls_name = first_value.__class__.__name__
            else:
                cls_name = ""

            cols = defaultdict(list)
            for item in iter(self):
                # unpack dict
                if isinstance(self, dict):
                    cols["key"].append(item)
                    item = self.__getitem__(item)

                # get id
                id_ = getattr(item, "id", None)
                if id_ is not None:
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
                                res = re.search("\[[+-]?\d+\]", attr)
                                has_index = False
                                if res:
                                    has_index = True
                                    index_str = res.group()
                                    index = int(
                                        index_str.replace("[", "").replace("]", "")
                                    )
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
                                        x.__repr_syft_nested__()
                                        if hasattr(x, "__repr_syft_nested__")
                                        else x
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

            # if custom_repr:
            table_icon = None
            if hasattr(values[0], "icon"):
                table_icon = values[0].icon
            # this is a list of dicts
            return create_table_template(
                df.to_dict("records"),
                f"{cls_name} {self.__class__.__name__.capitalize()}",
                table_icon=table_icon,
            )

    except Exception as e:
        print(f"error representing {type(self)} of objects. {e}")
        pass

    # stdlib
    import html

    return html.escape(self.__repr__())


# give lists and dicts a _repr_html_ if they contain SyftObject's
aggressive_set_attr(type([]), "_repr_html_", list_dict_repr_html)
aggressive_set_attr(type({}), "_repr_html_", list_dict_repr_html)


class StorableObjectType:
    def to(self, projection: type, context: Optional[Context] = None) -> Any:
        # 🟡 TODO 19: Could we do an mro style inheritence conversion? Risky?
        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self, context)


class PartialSyftObject(SyftObject, metaclass=PartialModelMetaclass):
    """Syft Object to which partial arguments can be provided."""

    __canonical_name__ = "PartialSyftObject"
    __version__ = SYFT_OBJECT_VERSION_1

    def __init__(self, *args, **kwargs) -> None:
        # Filter out Empty values from args and kwargs
        args_, kwargs_ = (), {}
        for arg in args:
            if arg is not Empty:
                args_.append(arg)

        for key, val in kwargs.items():
            if val is not Empty:
                kwargs_[key] = val

        super().__init__(*args_, **kwargs_)

        fields_with_default = set()
        for _field_name, _field in self.__fields__.items():
            if _field.default or _field.allow_none:
                fields_with_default.add(_field_name)

        # Fields whose values are set via a validator hook
        fields_set_via_validator = []

        for _field_name in self.__validators__.keys():
            _field = self.__fields__[_field_name]
            if self.__dict__[_field_name] is None:
                # Since all fields are None, only allow None
                # where either none is allowed or default is None
                if _field.allow_none or _field.default is None:
                    fields_set_via_validator.append(_field)

        # Exclude unset fields
        unset_fields = (
            set(self.__fields__)
            - set(self.__fields_set__)
            - set(fields_set_via_validator)
        )

        empty_fields = unset_fields - fields_with_default
        for field_name in empty_fields:
            self.__dict__[field_name] = Empty


recursive_serde_register_type(PartialSyftObject)


def attach_attribute_to_syft_object(result: Any, attr_dict: Dict[str, Any]) -> Any:
    box_to_result_type = None

    if type(result) in OkErr:
        box_to_result_type = type(result)
        result = result.value

    single_entity = False
    is_tuple = isinstance(result, tuple)

    if isinstance(result, (list, tuple)):
        iterable_keys = range(len(result))
        result = list(result)
    elif isinstance(result, dict):
        iterable_keys = result.keys()
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
    wrapped_result = tuple(wrapped_result) if is_tuple else wrapped_result
    if box_to_result_type is not None:
        wrapped_result = box_to_result_type(wrapped_result)

    return wrapped_result
