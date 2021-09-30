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


class Entity(Base):
    __tablename__ = "entity"

    name = Column(String(255), primary_key=True)
    entity_bin = Column(LargeBinary(3072))

    @property
    def obj(self) -> Any:
        return deserialize(self.entity_bin, from_bytes=True)  # TODO: techdebt fix

    @obj.setter
    def obj(self, value: Any) -> None:
        self.entity_bin = serialize(value, to_bytes=True)  # TODO: techdebt fix
