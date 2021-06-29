
from . import Base
from sqlalchemy import Column, Integer, String, ForeignKey

class BinObjDataset(Base):
    __tablename__ = "bin_obj_dataset"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    name = Column(String(256))
    obj = Column(String(256), ForeignKey("bin_object.id"))
    dataset = Column(String(256), ForeignKey("dataset.id"))
    dtype = Column(String(256))
    shape = Column(String(256))
