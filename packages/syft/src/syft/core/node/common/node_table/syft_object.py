# stdlib
from collections import defaultdict
import inspect
from inspect import Signature
from typing import Any
from typing import Callable
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

# third party
import pydantic
from pydantic import BaseModel
from pydantic import EmailStr
from pydantic.fields import Undefined
from typeguard import check_type

# relative
from .....lib.util import full_name_with_qualname
from .....lib.util import get_qualname_for
from ....common import UID
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serialize import _serialize as serialize
from ...new.credentials import SyftVerifyKey

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


class SyftObjectRegistry:
    __object_version_registry__: Dict[str, Type["SyftObject"]] = {}
    __object_transform_registry__: Dict[str, Callable] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__canonical_name__") and hasattr(cls, "__version__"):
            mapping_string = f"{cls.__canonical_name__}_{cls.__version__}"
            if mapping_string in cls.__object_version_registry__:
                raise Exception(f"Duplicate mapping for {mapping_string} and {cls}")
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
        if issubclass(type_from, SyftBaseObject):
            klass_from = type_from.__canonical_name__
            version_from = type_from.__version__
        else:
            klass_from = type_from.__name__
            version_from = None
        if issubclass(type_to, SyftBaseObject):
            klass_to = type_to.__canonical_name__
            version_to = type_to.__version__
        else:
            klass_to = type_to.__name__
            version_to = None

        mapping_string = f"{klass_from}_{version_from}_x_{klass_to}_{version_to}"
        return cls.__object_transform_registry__[mapping_string]


print_type_cache = defaultdict(list)


class SyftObject(SyftBaseObject, SyftObjectRegistry):
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

    __attr_state__: List[str]  # persistent recursive serde keys
    __attr_searchable__: List[str]  # keys which can be searched in the ORM
    __attr_unique__: List[
        str
    ]  # the unique keys for the particular Collection the objects will be stored in
    __serde_overrides__: Dict[
        str, Sequence[Callable]
    ] = {}  # List of attributes names which require a serde override.
    __owner__: str

    def to_mongo(self) -> Dict[str, Any]:
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
        # _repr_str += "\n"
        # fqn = full_name_with_qualname(type(self))
        # _repr_str += f'fqn = "{fqn}"\n'
        # _repr_str += f"mro = {[t.__name__ for t in type(self).mro()]}"

        # _repr_str += "\n\ncallables = [\n"
        # for func, sig in self.__syft_get_funcs__():
        #     _repr_str += f"  {func}{sig}: pass\n"
        # _repr_str += f"]"
        # return _repr_str

    def _repr_markdown_(self) -> str:
        text_repr = self.__repr__()
        return "```python\n" + text_repr + "\n```"

    @staticmethod
    def from_mongo(bson: Any) -> "SyftObject":
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
    def to(self, projection: type) -> Any:
        # 🟡 TODO 19: Could we do an mro style inheritence conversion? Risky?
        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self)

    def to_dict(self) -> Dict[str, Any]:
        # 🟡 TODO 18: Remove to_dict and replace usage with transforms etc
        return dict(self)

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
            type_ = cls.__fields__[key].type_
            # EmailStr seems to be lost every time the value is set even with a validator
            # this means the incoming type is str so our validators fail
            if issubclass(type_, EmailStr):
                type_ = str
            kt_dict[key] = type_
        return kt_dict

    @classmethod
    def _syft_unique_keys_dict(cls) -> Dict[str, type]:
        return cls._syft_keys_types_dict("__attr_unique__")

    @classmethod
    def _syft_searchable_keys_dict(cls) -> Dict[str, type]:
        return cls._syft_keys_types_dict("__attr_searchable__")


def transform_method(
    klass_from: Union[type, str],
    klass_to: Union[type, str],
    version_from: Optional[int] = None,
    version_to: Optional[int] = None,
) -> Callable:
    klass_from_str = (
        klass_from if isinstance(klass_from, str) else klass_from.__canonical_name__
    )
    klass_to_str = (
        klass_to if isinstance(klass_to, str) else klass_to.__canonical_name__
    )
    version_from = (
        version_from if isinstance(version_from, int) else klass_from.__version__
    )
    version_to = version_to if isinstance(version_to, int) else klass_to.__version__

    def decorator(function: Callable):
        SyftObjectRegistry.add_transform(
            klass_from=klass_from_str,
            version_from=version_from,
            klass_to=klass_to_str,
            version_to=version_to,
            method=function,
        )

        return function

    return decorator


def transform(
    klass_from: Union[type, str],
    klass_to: Union[type, str],
    version_from: Optional[int] = None,
    version_to: Optional[int] = None,
) -> Callable:
    if isinstance(klass_from, str):
        klass_from_str = klass_from

    if issubclass(klass_from, SyftBaseObject):
        klass_from_str = klass_from.__canonical_name__
        version_from = klass_from.__version__

    if not issubclass(klass_from, SyftBaseObject):
        klass_from_str = klass_from.__name__
        version_from = None

    if isinstance(klass_to, str):
        klass_to_str = klass_to

    if issubclass(klass_to, SyftBaseObject):
        klass_to_str = klass_to.__canonical_name__
        version_to = klass_to.__version__

    if not issubclass(klass_to, SyftBaseObject):
        klass_to_str = klass_to.__name__
        version_to = None

    def decorator(function: Callable):
        transforms = function()

        def wrapper(self: klass_from) -> klass_to:
            output = dict(self)
            for transform in transforms:
                output = transform(self, output)
            return klass_to(**output)

        SyftObjectRegistry.add_transform(
            klass_from=klass_from_str,
            version_from=version_from,
            klass_to=klass_to_str,
            version_to=version_to,
            method=wrapper,
        )

        return function

    return decorator
