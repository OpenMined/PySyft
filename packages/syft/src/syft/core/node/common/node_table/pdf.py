# stdlib
from typing import Any

# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import LargeBinary

# relative
from . import Base
from ..... import deserialize
from ..... import serialize


class PDFObject(Base):
    __tablename__ = "daa_pdf"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    binary = Column(LargeBinary(3072))
