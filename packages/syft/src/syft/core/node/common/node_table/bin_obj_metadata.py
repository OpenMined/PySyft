# stdlib
from typing import List
from typing import Optional

# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base
from .syft_object import SyftObject


class ObjectMetadata(Base):
    __tablename__ = "obj_metadata"

    __mapper_args__ = {"confirm_deleted_rows": False}

    id = Column(Integer(), primary_key=True, autoincrement=True)
    obj = Column(String(256))
    tags = Column(JSON())
    description = Column(String())
    name = Column(String())
    read_permissions = Column(JSON())
    search_permissions = Column(JSON())
    write_permissions = Column(JSON())
    is_proxy_dataset = Column(Boolean(), default=False)


class NoSQLObjectMetadata(SyftObject):
    # version
    __canonical_name__ = "ObjectMetadata"
    __version__ = 1

    # fields
    obj: str
    tags: List[str] = []
    description: str
    name: Optional[str]
    read_permissions: str
    search_permissions: str
    write_permissions: str
    is_proxy_dataset: bool = False

    # serde / storage rules
    __attr_state__ = [
        "obj"
        "tags"
        "description"
        "name"
        "read_permissions"
        "search_permissions"
        "write_permissions"
        "is_proxy_dataset"
    ]
    __attr_searchable__: List[str] = ["obj"]
    __attr_unique__: List[str] = []
