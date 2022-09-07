# stdlib
from typing import Any
from typing import Dict
from typing import List

# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base
from .user import SyftObject


class SetupConfig(Base):
    __tablename__ = "setup"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    domain_name = Column(String(255), default="")
    description = Column(String(255), default="")
    contact = Column(String(255), default="")
    daa = Column(Boolean(), default=False)
    node_id = Column(String(32), default="")
    daa_document = Column(Integer, ForeignKey("daa_pdf.id"))
    tags = Column(String(255), default="[]")
    deployed_on = Column(DateTime())
    signing_key = Column(String(2048))

    def __str__(self) -> str:
        return f"<Domain Name: {self.domain_name}>"


class NoSQLSetup(SyftObject):
    # version
    __canonical_name__ = "Setup"
    __version__ = 1

    # fields
    id_int: int
    domain_name: str = ""
    description: str = ""
    contact: str = ""
    daa: bool = False
    node_uid: str = ""
    daa_document: bytes = b""
    tags: List[str] = []
    deployed_on: str
    signing_key: str

    # serde / storage rules
    __attr_state__ = [
        "id_int",
        "domain_name",
        "description",
        "contact" "daa" "node_uid",
        "daa_document" "tags",
        "deployed_on",
        "signing_key",
    ]
    __attr_searchable__ = ["node_uid", "id_int", "signing_key", "domain_name"]
    __attr_unique__ = ["node_uid"]

    def to_dict(self) -> Dict[Any, Any]:
        attr_dict = super().to_dict()
        del attr_dict["id"]
        del attr_dict["id_int"]
        del attr_dict["daa_document"]
        return attr_dict
