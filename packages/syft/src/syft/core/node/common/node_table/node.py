# stdlib
from typing import List
from typing import Optional

# relative
from .syft_object import SyftObject


class NoSQLNodeRoute(SyftObject):
    # version
    __canonical_name__ = "NodeRoute"
    __version__ = 1

    # fields
    host_or_ip: str
    is_vpn: bool = False
    private: bool = False
    protocol: str = "http"
    port: int = 80
    vpn_endpoint: str = ""
    vpn_key: str = ""

    # serde / storage rules
    __attr_state__ = [
        "host_or_ip",
        "is_vpn",
        "private",
        "protocol",
        "port",
        "vpn_endpoint",
        "vpn_key",
    ]

    __attr_searchable__ = ["host_or_ip"]
    __attr_unique__ = ["host_or_ip"]


class NoSQLNode(SyftObject):
    # version
    __canonical_name__ = "Node"
    __version__ = 1

    # fields
    node_uid: str
    node_name: str
    node_type: Optional[str]
    verify_key: Optional[str]
    keep_connected: Optional[bool] = True
    node_route: List[NoSQLNodeRoute] = []

    # serde / storage rules
    __attr_state__ = [
        "node_uid",
        "node_name",
        "node_type",
        "verify_key",
        "keep_connected",
        "node_route",
    ]

    __attr_searchable__ = ["node_uid", "verify_key"]
    __attr_unique__ = ["node_uid"]
