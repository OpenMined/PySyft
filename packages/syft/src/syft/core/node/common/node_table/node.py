# stdlib
from typing import Optional, List

# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base
from .user import SyftObject


# this table holds the list of known nodes usually peer domains
class Node(Base):
    __tablename__ = "node"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    node_uid = Column(String(255))
    node_name = Column(String(255))
    node_type = Column(String(255))
    verify_key = Column(String(2048))
    keep_connected = Column(Boolean(), default=True)


class NoSQLNode(SyftObject):
    # version
    __canonical_name__ = "Node"
    __version__ = 1

    # fields
    id_int: int
    node_uid: str
    node_name: str
    node_type: Optional[str]
    verify_key: Optional[str]
    keep_connected: Optional[bool] = True
    node_route: List[dict] = []
    # host_or_ip: Optional[str]
    # is_vpn: Optional[bool] = False
    # protocol: Optional[str] = "http"
    # port: Optional[int] = 80
    # private: Optional[bool] = False
    # vpn_endpoint: Optional[str] = ""
    # vpn_key: Optional[str] = ""

    # serde / storage rules
    __attr_state__ = [
        "id_int",
        "node_uid",
        "node_name",
        "node_type",
        "verify_key",
        "keep_connected",
        "node_route",
        # "host_or_ip",
        # "is_vpn",
        # "private",
        # "protocol",
        # "port",
        # "vpn_endpoint",
        # "vpn_key",
    ]

    __attr_searchable__ = ["node_uid", "verify_key", "id_int"]
    __attr_unique__ = ["node_uid"]
