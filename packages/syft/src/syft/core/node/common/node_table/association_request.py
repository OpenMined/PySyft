# stdlib
from typing import Dict

# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.types import LargeBinary

# relative
from . import Base
from ..... import deserialize
from ...abstract.node import AbstractNodeClient


class AssociationRequest(Base):
    """Association Request.

    Columns:
        id (Integer, Primary Key): Cycle ID.
        date (TIME): Start time.
        name (String): Organization / Domain name.
        address (String) : Organization / Domain Address.
        accepted (Bool) :  If request was accepted or not.
        pending (Bool) : If association request is pending.
        handshake_value (String) : Association request unique identifier
    """

    __tablename__ = "association_request"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requested_date = Column(String(255))
    accepted_date = Column(String(255), default="")
    address = Column(String(255), default="")
    node = Column(String(255), default="")
    name = Column(String(255), default="")
    email = Column(String(255), default="")
    reason = Column(String(255), default="")
    status = Column(String(255), default="")
    source = Column(LargeBinary(4096), default=b"")
    target = Column(LargeBinary(4096), default=b"")

    def __str__(self) -> str:
        return (
            f"< Association Request id : {self.id}, Name: {self.name},"
            f" Address: {self.address} , status: {self.status}, Date: {self.requested_date}>"
        )

    def get_metadata(self) -> Dict[str, str]:
        return {
            "requested_date": self.requested_date,
            "name": self.name,
            "email": self.email,
            "reason": self.reason,
            "status": self.status,
        }

    def get_source(self) -> AbstractNodeClient:
        return deserialize(self.source, from_bytes=True)

    def get_target(self) -> AbstractNodeClient:
        return deserialize(self.target, from_bytes=True)
