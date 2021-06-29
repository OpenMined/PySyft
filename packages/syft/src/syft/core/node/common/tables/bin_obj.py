# stdlib
from typing import Any

# third party
from sqlalchemy import Column
from sqlalchemy import LargeBinary
from sqlalchemy import String
from sqlalchemy.orm.decl_api import DeclarativeMeta

# syft absolute
from syft import deserialize
from syft import serialize

from . import Base
from sqlalchemy import Column, Integer, String, ForeignKey

class BinObject(Base):  # type: ignore
    # __tablename__ = "bin_object"

    id = Column(String(3072), primary_key=True)
    binary = Column(LargeBinary(3072))
    obj_name = Column(String(3072))

    @property
    def object(self) -> Any:
        # storing DataMessage, we should unwrap
        return deserialize(self.binary, from_bytes=True)  # TODO: techdebt fix

    @object.setter
    def object(self, value: Any) -> None:
        # storing DataMessage, we should unwrap
        self.binary = serialize(value, to_bytes=True)  # TODO: techdebt fix
        # self.obj_name = type(value).__name__

