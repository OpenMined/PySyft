# stdlib
from typing import Any

# third party
from sqlalchemy import Column
from sqlalchemy import LargeBinary
from sqlalchemy import String

# relative
from . import Base
from ..... import deserialize
from ..... import serialize


class BinObject(Base):
    __tablename__ = "bin_object"

    id = Column(String(256), primary_key=True)
    binary = Column(LargeBinary(3072))
    obj_name = Column(String(3072))

    @property
    def obj(self) -> Any:
        # storing DataMessage, we should unwrap
        return deserialize(self.binary, from_bytes=True)  # TODO: techdebt fix

    @obj.setter
    def obj(self, value: Any) -> None:
        # storing DataMessage, we should unwrap
        self.binary = serialize(value, to_bytes=True)  # TODO: techdebt fix
