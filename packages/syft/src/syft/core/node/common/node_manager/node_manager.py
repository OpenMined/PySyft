# stdlib
from typing import Any
from typing import Dict
from typing import Optional, List

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine

# relative
from .....grid import GridURL
from ..node_service.node_credential.node_credentials import NodeCredentials
from ..node_service.node_route.route_update import RouteUpdate
from ..node_table.node import NoSQLNode
from ..node_table.node import Node
from .database_manager import DatabaseManager
from .database_manager import NoSQLDatabaseManager


class NodeNotFoundError(Exception):
    pass


class NodeManager(DatabaseManager):
    schema = Node

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=NodeManager.schema, db=database)

    def create_or_get_node(self, node_uid: str, node_name: str) -> int:
        # node_uid is a UID as a string with no_dash
        node = self.first(node_uid=node_uid)
        if node:
            self.modify(query={"node_uid": node_uid}, values={"node_name": node_name})
        else:
            self.register(**{"node_uid": node_uid, "node_name": node_name})
            node = self.first(node_uid=node_uid)
        return node.id

    def add_or_update_node_credentials(self, credentials: NodeCredentials) -> None:
        node = self.first(node_uid=credentials.node_uid)
        if node:
            if node.verify_key is not None:
                credentials.validate(key=node.verify_key)
            self.modify(
                query={"node_uid": credentials.node_uid}, values={**credentials}
            )
        else:
            self.register(**credentials)

    def validate_id_and_key(
        self, node_uid: str, verify_key: VerifyKey
    ) -> Optional[Node]:
        return self.first(
            node_uid=node_uid,
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8"),
        )

    def get_node_for(self, verify_key: VerifyKey) -> Optional[Node]:
        return self.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )


class NoSQLNodeManager(NoSQLDatabaseManager):
    """Class to manage node database actions."""

    _collection_name = "node"
    __canonical_object_name__ = "Node"

    def first(self, **kwargs: Any) -> NoSQLNode:
        result = super().find_one(kwargs)
        if not result:
            raise NodeNotFoundError
        return result

    def create_route_dict(
        self,
        host_or_ip: str,
        is_vpn: bool = False,
        private: bool = False,
        protocol: str = "http",
        port: int = 80,
        vpn_endpoint: str = "",
        vpn_key="",
    ):
        if host_or_ip is None:
            raise ValueError(f"Route addition required valid host_or_ip:{host_or_ip}")
        return {
            "host_or_ip": host_or_ip,
            "is_vpn": is_vpn,
            "private": private,
            "protocol": protocol,
            "port": port,
            "vpn_endpoint": vpn_endpoint,
            "vpn_key": vpn_key,
        }

    def create_or_get_node(
        self,
        node_uid: str,
        node_name: str,
        host_or_ip: str,
        is_vpn: Optional[bool] = False,
        vpn_endpoint: Optional[str] = "",
        vpn_key: Optional[str] = "",
    ) -> NoSQLNode:
        # node_uid is a UID as a string with no_dash
        try:
            node = self.first(node_uid=node_uid)
            attributes: Dict[str, Any] = {}

            _exists = False  # Flag to check if route already present.
            for route in node.node_route:
                if not "host_or_ip" in route.keys():
                    raise ValueError(
                        f"The route dict:{route} should have host_or_ip attribute."
                    )
                if route["host_or_ip"] == host_or_ip:
                    _exists = True
                    break

            if not _exists:
                new_route = self.create_route_dict(
                    host_or_ip=host_or_ip,
                    is_vpn=is_vpn,
                    vpn_endpoint=vpn_endpoint,
                    vpn_key=vpn_key,
                )
                node.node_route.append(new_route)

            attributes["__blob__"] = node.to_bytes()

            self.update_one(
                query={"node_uid": node_uid},
                values=attributes,
            )
        except NodeNotFoundError:
            curr_len = len(self)

            node_row = NoSQLNode(
                id_int=curr_len + 1,
                node_uid=node_uid,
                node_name=node_name,
            )
            new_route = self.create_route_dict(
                host_or_ip=host_or_ip,
                is_vpn=is_vpn,
                vpn_endpoint=vpn_endpoint,
                vpn_key=vpn_key,
            )
            node_row.node_route.append(new_route)
            self.add(node_row)

        return self.first(node_uid=node_uid)

    def add_or_update_node_credentials(self, credentials: NodeCredentials) -> None:
        node = self.first(node_uid=credentials.node_uid)
        credentials_dict: Dict[str, Any] = {**credentials}
        if node:
            if node.verify_key is not None:
                credentials.validate(key=node.verify_key)
            attributes = {}
            for k, v in credentials_dict.items():
                if k not in node.__attr_state__:
                    raise ValueError(f"Cannot set an non existing field:{k} to Node")
                else:
                    setattr(node, k, v)
                if k in node.__attr_searchable__:
                    attributes[k] = v
            attributes["__blob__"] = node.to_bytes()

            self.update_one(query={"node_uid": credentials.node_uid}, values=attributes)
        else:
            curr_len = len(self)
            node_row = NoSQLNode(
                id_int=curr_len + 1,
                **credentials_dict,
            )
            self.add(node_row)

    def validate_id_and_key(self, node_uid: str, verify_key: VerifyKey) -> NoSQLNode:
        return self.first(
            node_uid=node_uid,
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8"),
        )

    def get_node_for(self, verify_key: VerifyKey) -> NoSQLNode:
        return self.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )

    
    def validate_route_update(
        self,
        node_collection: List[NoSQLNode],
        curr_node: NoSQLNode,
        route_update: RouteUpdate,
    ) -> bool:
        "Valid if the input route is not assigned to any other node than the current node."
        if not route_update.source_node_url:
            raise Exception("source_node_url is missing")
        source_url = GridURL.from_url(route_update.source_node_url)

        host_or_ip = source_url.host_or_ip
        port = source_url.port
        _valid = True  # Initial flag assuming that the route does not exists
        for node in node_collection:
            if node.node_uid == curr_node.node_uid:
                continue
            for route in node.node_route:
                if not "host_or_ip" in route.keys() or not "port" in route.keys() :
                    raise ValueError(
                        f"The route dict:{route} should have host_or_ip and port attribute"
                    )
                if host_or_ip == route["host_or_ip"] and port == route["port"]:
                    _valid = False
                    break
            if not _valid:
                break

        return _valid

    def update_route(
        self, curr_node: NoSQLNode, route_update: RouteUpdate, is_vpn: bool = False
    ) -> None:
        if not route_update.source_node_url:
            raise Exception("source_node_url is missing")
        source_url = GridURL.from_url(route_update.source_node_url)

        new_route = self.create_route_dict(
            host_or_ip=source_url.host_or_ip,
            protocol=source_url.protocol,
            port=source_url.port,
            private=route_update.private,
            is_vpn=is_vpn,
        )
        route_index = -1 #Stores the index of the route with the above host_or_ip
        try:
            node = self.first(
                 node_uid=curr_node.node_uid
            )
            for idx, route in enumerate(node.node_route):
                if not "host_or_ip" in route.keys():
                    raise ValueError(
                        f"The route dict:{route} should have host_or_ip attribute."
                    )
                if route["host_or_ip"] == source_url.host_or_ip:
                    route_index = idx
                    break
            if route_index==-1: #route does not exists add new route
                curr_node.node_route.append(new_route)
            else:
                curr_node.node_route[route_index] = new_route

            attributes = {}
            attributes["__blob__"] = curr_node.to_bytes()

            self.update_one(
                query={
                    "node_uid": curr_node.node_uid,
                },
                values=attributes,
            )

        except NodeNotFoundError:
            raise NodeNotFoundError(f"Update Route does not have valid node to update with uid: {curr_node.node_uid}")
    

    def get_routes(self, node_row: NoSQLNode) -> List[Dict]:
        return node_row.node_route
