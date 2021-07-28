# third party
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base


class ObjectMetadata(Base):
    __tablename__ = "obj_metadata"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    obj = Column(String(256), ForeignKey("bin_object.id", ondelete="CASCADE"))
    tags = Column(JSON())
    description = Column(String())
    name = Column(String())
    read_permissions = Column(JSON())
    search_permissions = Column(JSON())
