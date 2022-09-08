# stdlib
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Sequence
from typing import Type

# third party
import pydantic
from pydantic import BaseModel

# relative
from .....lib.python import Dict as SyDict
from ....common import UID
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serialize import _serialize as serialize


class SyftObjectRegistry:
    __object_version_registry__: Dict[str, Dict[int, Type["SyftObject"]]] = defaultdict(
        lambda: {}
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__canonical_name__"):
            cls.__object_version_registry__[cls.__canonical_name__][  # type: ignore
                int(cls.__version__)  # type: ignore
            ] = cls  # type: ignore

    @classmethod
    def versioned_class(cls, name: str, version: int) -> Optional[Type["SyftObject"]]:
        if name not in cls.__object_version_registry__:
            return None
        classes = cls.__object_version_registry__[name]
        if version not in classes:
            return None
        return classes[version]


class SyftObject(BaseModel, SyftObjectRegistry):
    class Config:
        arbitrary_types_allowed = True

    # all objects have a UID
    id: Optional[UID] = None  # consistent and persistent uuid across systems

    @pydantic.validator("id", pre=True, always=True)
    def make_id(cls, v: Optional[UID]) -> UID:
        return v if isinstance(v, UID) else UID()

    __canonical_name__: str  # the name which doesn't change even when there are multiple classes
    __version__: int  # data is always versioned
    __attr_state__: List[str]  # persistent recursive serde keys
    __attr_searchable__: List[str]  # keys which can be searched in the ORM
    __attr_unique__: List[
        str
    ]  # the unique keys for the particular Collection the objects will be stored in
    __serde_overrides__: Dict[
        str, Sequence[Callable]
    ] = {}  # List of attributes names which require a serde override.

    def to_mongo(self) -> Dict[str, Any]:
        d = {}
        for k in self.__attr_searchable__:
            d[k] = getattr(self, k)
        blob = self.to_bytes()
        d["_id"] = self.id.value  # type: ignore
        d["__canonical_name__"] = self.__canonical_name__
        d["__version__"] = self.__version__
        d["__blob__"] = blob

        return d

    def to_dict(self) -> Dict[Any, Any]:
        attr_dict = dict(**self)
        return attr_dict

    def to_bytes(self) -> bytes:
        d = SyDict(**self)
        for attr, funcs in self.__serde_overrides__.items():
            if attr in d:
                d[attr] = funcs[0](d[attr])
        return serialize(d, to_bytes=True)  # type: ignore

    @staticmethod
    def from_bytes(blob: bytes) -> "SyftObject":
        return deserialize(blob, from_bytes=True)

    @staticmethod
    def from_mongo(bson: Any) -> "SyftObject":
        constructor = SyftObjectRegistry.versioned_class(
            name=bson["__canonical_name__"], version=bson["__version__"]
        )
        if constructor is None:
            raise ValueError(
                "Versioned class should not be None for initialization of SyftObject."
            )
        de = deserialize(bson["__blob__"], from_bytes=True).upcast()
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
