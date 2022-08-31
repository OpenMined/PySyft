# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Type

# third party
import pydantic
from pydantic import BaseModel
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# syft absolute
import syft as sy

# relative
from . import Base
from .....lib.python import Dict as SyDict
from ....common import UID


class UserApplication(Base):
    __tablename__ = "syft_application"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    email = Column(String(255))
    name = Column(String(255), default="")
    hashed_password = Column(String(512))
    salt = Column(String(255))
    daa_pdf = Column(Integer, ForeignKey("daa_pdf.id"))
    status = Column(String(255), default="pending")
    added_by = Column(String(2048))
    website = Column(String(2048))
    institution = Column(String(2048))
    budget = Column(Float(), default=0.0)

    def __str__(self) -> str:
        return (
            f"<User Application id: {self.id}, email: {self.email}, name: {self.name}"
            f"status: {self.status}>"
        )


class SyftUser(Base):
    __tablename__ = "syft_user"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    email = Column(String(255))
    name = Column(String(255), default="")
    budget = Column(Float(), default=0.0)
    hashed_password = Column(String(512))
    salt = Column(String(255))
    private_key = Column(String(2048))
    verify_key = Column(String(2048))
    role = Column(Integer, ForeignKey("role.id"))
    added_by = Column(String(2048))
    website = Column(String(2048))
    institution = Column(String(2048))
    daa_pdf = Column(Integer, ForeignKey("daa_pdf.id"))
    created_at = Column(DateTime())

    def __str__(self) -> str:
        return (
            f"<User id: {self.id}, email: {self.email}, name: {self.name}"
            f"role: {self.role}>"
        )


def create_user(
    email: str,
    hashed_password: str,
    salt: str,
    private_key: str,
    role: int,
    name: str = "",
    budget: float = 0.0,
) -> SyftUser:
    new_user = SyftUser(
        email=email,
        hashed_password=hashed_password,
        salt=salt,
        private_key=private_key,
        role=role,
        name=name,
        budget=budget,
    )
    return new_user


class SyftObjectRegistry:
    __object_version_registry__: Dict[str, Dict[int, Type[SyftObject]]] = defaultdict(
        lambda: {}
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__canonical_name__"):
            cls.__object_version_registry__[cls.__canonical_name__][  # type: ignore
                int(cls.__version__)  # type: ignore
            ] = cls  # type: ignore

    @classmethod
    def versioned_class(cls, name: str, version: int) -> Optional[Type[SyftObject]]:
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

    def to_bytes(self) -> bytes:
        d = SyDict(**self)
        return sy.serialize(d, to_bytes=True)

    @staticmethod
    def from_bytes(blob: bytes) -> SyftObject:
        return sy.deserialize(blob, from_bytes=True)

    @staticmethod
    def from_mongo(bson: Any) -> SyftObject:
        constructor = SyftObjectRegistry.versioned_class(
            name=bson["__canonical_name__"], version=bson["__version__"]
        )
        if constructor is None:
            raise ValueError(
                "Versioned class should not be None for initialization of SyftObject."
            )
        return constructor(**sy.deserialize(bson["__blob__"], from_bytes=True).upcast())

    # allows splatting with **
    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    # allows splatting with **
    def __getitem__(self, key: str) -> Any:
        return self.__dict__.__getitem__(key)

    def _upgrade_version(self, latest: bool = True) -> SyftObject:
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


class NoSQLSyftUser(SyftObject):
    # version
    __canonical_name__ = "SyftUser"
    __version__ = 1

    # fields
    email: str
    name: str
    budget: float
    hashed_password: str
    salt: str
    private_key: str
    verify_key: Optional[str]
    role: dict
    added_by: Optional[str]
    website: Optional[str]
    institution: Optional[str]
    daa_pdf: Optional[str]
    created_at: Optional[str]
    id_int: Optional[int]

    # serde / storage rules
    __attr_state__ = [
        "email",
        "name",
        "budget",
        "hashed_password",
        "salt",
        "private_key",
        "verify_key",
        "role",
        "added_by",
        "website",
        "institution",
        "daa_pdf",
        "created_at",
        "id_int",
    ]
    __attr_searchable__ = ["email", "verify_key", "id_int"]
    __attr_unique__ = ["email"]


# class NoSQLUserApplication(SyftObject):
#     # version
#     __canonical_name__ = "UserApplication"
#     __version__ = 1

#     # fields
#     email: str
#     name: str
#     hashed_password: str
#     salt: str
#     daa_pdf: Optional[int]
#     status: str
#     added_by: Optional[str]
#     website: Optional[str]
#     institution: Optional[str]
#     budget: float
#     id_int: int

#     # serde / storage rules
#     __attr_state__ = [
#         "email",
#         "name",
#         "budget",
#         "hashed_password",
#         "salt",
#         "added_by",
#         "website",
#         "institution",
#         "daa_pdf",
#         "id_int",
#     ]
#     __attr_searchable__ = ["email","id_int","status"]
#     __attr_unique__ = ["email"]
