# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base
from .user import SyftObject


class AssociationRequest(Base):
    __tablename__ = "association_request"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requested_date = Column(String(255))
    accepted_date = Column(String(255), default="")
    node_name = Column(String(255), default="")
    node_address = Column(String(255), default="")
    name = Column(String(255), default="")
    email = Column(String(255), default="")
    reason = Column(String(255), default="")
    status = Column(String(255), default="")
    source = Column(String(255), default="")
    target = Column(String(255), default="")

    def __str__(self) -> str:
        return (
            f"< Association Request id : {self.id}, Name: {self.name},"
            f" Status: {self.status}, Source: {self.source}, Target: {self.target}"
            f" Date: {self.requested_date}>"
        )

    def get_metadata(self) -> Dict[str, str]:
        return {
            "association_id": str(self.id),
            "requested_date": self.requested_date,
            "name": self.name,
            "email": self.email,
            "reason": self.reason,
            "status": self.status,
            "source": self.source,
            "target": self.target,
            "node_name": self.node_name,
            "node_address": self.node_address,
        }


class NoSQLAssociationRequest(SyftObject):
    # version
    __canonical_name__ = "AssociationRequest"
    __version__ = 1

    # fields
    id_int: int
    requested_date: str
    processed_date: str = ""
    node_name: str = ""
    node_address: str = ""
    name: str = ""
    email: str = ""
    reason: Optional[str] = ""
    status: str = ""
    source: str = ""
    target: str = ""

    # serde / storage rules
    __attr_state__ = [
        "id_int",
        "requested_date",
        "processed_date",
        "node_name",
        "node_address",
        "name",
        "email",
        "reason",
        "status",
        "source",
        "target",
    ]
    __attr_searchable__ = ["source", "target", "id_int"]
    __attr_unique__ = []

    def to_dict(self) -> Dict[Any, Any]:
        attr_dict = super().to_dict()
        del attr_dict["id"]
        attr_dict["association_id"] = str(attr_dict["id_int"])
        del attr_dict["id_int"]
        return attr_dict
