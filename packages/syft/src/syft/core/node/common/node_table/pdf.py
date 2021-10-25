# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import LargeBinary

# relative
from . import Base


class PDFObject(Base):
    __tablename__ = "daa_pdf"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    binary = Column(LargeBinary(3072))
