# stdlib
from collections import defaultdict
import inspect
from inspect import Signature
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
import warnings

# third party
import pydantic
from pydantic import BaseModel
from pydantic import EmailStr
from pydantic.fields import Undefined
from typeguard import check_type

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.deserialize import _deserialize as deserialize
from ..serde.recursive_primitives import recursive_serde_register_type
from ..serde.serialize import _serialize as serialize
from ..util.util import aggressive_set_attr
from ..util.util import full_name_with_qualname
from ..util.util import get_qualname_for
from .syft_metaclass import Empty
from .syft_metaclass import PartialModelMetaclass
from .uid import UID

SYFT_OBJECT_VERSION_1 = 1
SYFT_OBJECT_VERSION_2 = 2

supported_object_versions = [SYFT_OBJECT_VERSION_1, SYFT_OBJECT_VERSION_2]

HIGHEST_SYFT_OBJECT_VERSION = max(supported_object_versions)
LOWEST_SYFT_OBJECT_VERSION = min(supported_object_versions)


class SyftBaseObject(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    __canonical_name__: str  # the name which doesn't change even when there are multiple classes
    __version__: int  # data is always versioned


class Context(SyftBaseObject):
    pass


class SyftObjectRegistry:
    __object_version_registry__: Dict[str, Type["SyftObject"]] = {}
    __object_transform_registry__: Dict[str, Callable] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__canonical_name__") and hasattr(cls, "__version__"):
            mapping_string = f"{cls.__canonical_name__}_{cls.__version__}"
            if mapping_string in cls.__object_version_registry__:
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

    __attr_searchable__: List[str] = []  # keys which can be searched in the ORM
    __attr_unique__: List[str] = []
    # the unique keys for the particular Collection the objects will be stored in
    __serde_overrides__: Dict[
        str, Sequence[Callable]
    ] = {}  # List of attributes names which require a serde override.
    __owner__: str

    __attr_repr_cols__: List[str] = []  # show these in html repr collections

    def to_mongo(self) -> Dict[str, Any]:
        warnings.warn(
            "`SyftObject.to_mongo` is deprecated and will be removed in a future version",
            PendingDeprecationWarning,
        )

        d = {}
        for k in self.__attr_searchable__:
            # 🟡 TODO 24: pass in storage abstraction and detect unsupported types
            # if unsupported, convert to string
            value = getattr(self, k, "")
            if isinstance(value, SyftVerifyKey):
                value = str(value)
            d[k] = value
        blob = serialize(dict(self), to_bytes=True)
        d["_id"] = self.id.value  # type: ignore
        d["__canonical_name__"] = self.__canonical_name__
        d["__version__"] = self.__version__
        d["__blob__"] = blob
        d["__repr__"] = self.__repr__()

        return d

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
            value = getattr(self, attr, "<Missing>")
            value_type = full_name_with_qualname(type(attr))
            value_type = value_type.replace("builtins.", "")
            value = f'"{value}"' if isinstance(value, str) else value
            _repr_str += f"  {attr}: {value_type} = {value}\n"
        return _repr_str

    def _repr_markdown_(self) -> str:
        class_name = get_qualname_for(type(self))
        _repr_str = f"class {class_name}:\n"
        fields = getattr(self, "__fields__", {})
        for attr in fields.keys():
            value = getattr(self, attr, "<Missing>")
            value_type = full_name_with_qualname(type(attr))
            value_type = value_type.replace("builtins.", "")
            value = f'"{value}"' if isinstance(value, str) else value
            _repr_str += f"  {attr}: {value_type} = {value}\n"

        # _repr_str += "\n"
        # fqn = full_name_with_qualname(type(self))
        # _repr_str += f'fqn = "{fqn}"\n'
        # _repr_str += f"mro = {[t.__name__ for t in type(self).mro()]}"

        # _repr_str += "\n\ncallables = [\n"
        # for func, sig in self.__syft_get_funcs__():
        #     _repr_str += f"  {func}{sig}: pass\n"
        # _repr_str += f"]"
        # return _repr_str
        return "```python\n" + _repr_str + "\n```"

    @staticmethod
    def from_mongo(bson: Any) -> "SyftObject":
        warnings.warn(
            "`SyftObject.from_mongo` is deprecated and will be removed in a future version",
            PendingDeprecationWarning,
        )

        constructor = SyftObjectRegistry.versioned_class(
            name=bson["__canonical_name__"], version=bson["__version__"]
        )
        if constructor is None:
            raise ValueError(
                "Versioned class should not be None for initialization of SyftObject."
            )
        de = deserialize(bson["__blob__"], from_bytes=True)
        for attr, funcs in constructor.__serde_overrides__.items():
            if attr in de:
                de[attr] = funcs[1](de[attr])
        return constructor(**de)

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
            return dict(self)
        else:
            new_dict = {}
            for k, v in dict(self).items():
                if exclude_empty and v is not Empty:
                    new_dict[k] = v
                if exclude_none and v is not None:
                    new_dict[k] = v
            return new_dict

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


def list_dict_repr_html(self) -> str:
    try:
        max_check = 1
        items_checked = 0
        has_syft = False
        extra_fields = []
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
                extra_fields = getattr(item, "__attr_repr_cols__", [])
                break
        if has_syft:
            # third party
            import pandas as pd

            cols = defaultdict(list)
            max_lines = 5
            line = 0
            for item in iter(self):
                line += 1
                if line > max_lines:
                    break
                if isinstance(self, dict):
                    cols["key"].append(item)
                    item = self.__getitem__(item)

                if type(item) == type:
                    cols["type"].append(full_name_with_qualname(item))
                else:
                    cols["type"].append(item.__repr__())

                cols["id"].append(getattr(item, "id", None))
                for field in extra_fields:
                    value = getattr(item, field, None)
                    cols[field].append(value)

            x = pd.DataFrame(cols)
            collection_type = (
                f"{type(self).__name__.capitalize()} - Size: {len(self)}\n"
            )
            return collection_type + x._repr_html_()
    except Exception as e:
        print(e)
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

        # Exclude unset fields
        unset_fields = set(self.__fields__) - set(self.__fields_set__)

        empty_fields = unset_fields - fields_with_default
        for field_name in empty_fields:
            self.__dict__[field_name] = Empty


recursive_serde_register_type(PartialSyftObject)
