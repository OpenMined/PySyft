# stdlib
from typing import Any

# third party
from sqlalchemy import Column
from sqlalchemy import LargeBinary
from sqlalchemy import String
from sqlalchemy import Integer

# syft absolute
from syft import deserialize
from syft import serialize

# relative
from . import Base


class Mechanism(Base):
    __tablename__ = "mechanism"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    mechanism_bin = Column(LargeBinary(3072), default=None)

    @property
    def obj(self) -> Any:
        return deserialize(self.mechanism_bin, from_bytes=True)  # TODO: techdebt fix

    @obj.setter
    def obj(self, value: Any) -> None:
        self.mechanism_bin = serialize(value, to_bytes=True)  # TODO: techdebt fix
