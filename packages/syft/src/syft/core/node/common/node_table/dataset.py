# stdlib
from typing import Any
from typing import Union

# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import LargeBinary
from sqlalchemy import String

# relative
from . import Base


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(String(256), primary_key=True)
    name = Column(String(256))
    manifest = Column(String(2048))
    description = Column(String(2048))
    tags = Column(JSON())
    str_metadata = Column(JSON())
    blob_metadata = Column(JSON())
