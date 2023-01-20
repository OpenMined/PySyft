# third party
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class BinObjDataset(Base):
    __tablename__ = "bin_obj_dataset"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    name = Column(String(256))
    obj = Column(String(256))
    dataset = Column(String(256), ForeignKey("dataset.id", ondelete="CASCADE"))
    dtype = Column(String(256))
    shape = Column(String(256))
