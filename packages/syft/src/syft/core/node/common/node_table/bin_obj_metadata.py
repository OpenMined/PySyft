# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base


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
