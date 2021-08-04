# third party
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class DatasetGroup(Base):
    __tablename__ = "datasetgroup"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    bin_object = Column(String(), ForeignKey("bin_object.id"))
    dataset = Column(String(), ForeignKey("json_object.id"))

    def __str__(self) -> str:
        return (
            f"<DatasetGroup id: {self.id}, bin_object: {self.bin_object}, "
            f"dataset: {self.dataset}>"
        )
