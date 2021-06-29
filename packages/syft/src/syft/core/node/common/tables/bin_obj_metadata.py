# third party
from sqlalchemy import JSON

from . import Base
from sqlalchemy import Column, Integer, String, ForeignKey


class ObjectMetadata(Base):  # type: ignore
    __tablename__ = "obj_metadata"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    obj = Column(Integer(), ForeignKey("bin_object.id"))
    tags = Column(JSON())
    description = Column(String())
    name = Column(String())
    read_permissions = Column(JSON())
    search_permissions = Column(JSON())
