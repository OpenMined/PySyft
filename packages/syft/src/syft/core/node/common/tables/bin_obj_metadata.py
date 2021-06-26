# third party
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy.orm.decl_api import DeclarativeMeta


def BinObjectMetadata(TableBase: DeclarativeMeta) -> DeclarativeMeta:
    class ObjectMetadata(TableBase):  # type: ignore
        __tablename__ = "obj_metadata"

        id = Column(Integer(), primary_key=True, autoincrement=True)
        obj = Column(Integer, ForeignKey("bin_object.id"))
        tags = Column(JSON())
        description = Column(String())
        name = Column(String())
        read_permissions = Column(JSON())
        search_permissions = Column(JSON())

    return ObjectMetadata
