# stdlib
from typing import Any

# third party
from sqlalchemy import Column
from sqlalchemy import LargeBinary
from sqlalchemy import String
from sqlalchemy import Float

# syft absolute
from syft import deserialize
from syft import serialize

# relative
from . import Base


class Ledger(Base):
    __tablename__ = "ledger"

    id = Column(String(256), primary_key=True)
    entity_name = Column(String(256))
    mechanism_name = Column(String(256))
    max_budget = Column(Float())
    delta = Column(Float())
    mechanism_bin = Column(LargeBinary(3072))
    entity_bin = Column(LargeBinary(3072))

    @property
    def mechanism(self) -> Any:
        return deserialize(self.mechanism_bin, from_bytes=True)  # TODO: techdebt fix

    @mechanism.setter
    def mechanism(self, value: Any) -> None:
        self.mechanism_bin = serialize(value, to_bytes=True)  # TODO: techdebt fix

    @property
    def entity(self) -> Any:
        return deserialize(self.entity_bin, from_bytes=True)  # TODO: techdebt fix

    @entity.setter
    def entity(self, value: Any) -> None:
        self.entity_bin = serialize(value, to_bytes=True)  # TODO: techdebt fix
