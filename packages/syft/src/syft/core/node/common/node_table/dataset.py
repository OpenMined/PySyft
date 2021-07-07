# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import LargeBinary

# relative
from . import Base

from typing import Union
from typing import Any

class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(String(256), primary_key=True)
    name = Column(String(256))
    manifest = Column(String(2048))
    description = Column(String(2048))
    tags = Column(JSON())
    str_metadata = Column(JSON())
    blob_metadata = Column(JSON())