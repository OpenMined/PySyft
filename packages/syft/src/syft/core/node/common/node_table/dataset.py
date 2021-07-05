# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(String(256), primary_key=True)
    manifest = Column(String(2048))
    description = Column(String(2048))
    tags = Column(JSON())
