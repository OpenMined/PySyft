# stdlib
from typing import Any
from typing import Dict
from typing import Optional

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


class NoSQLUserManager(NoSQLDatabaseManager):
    """Class to manage node database actions."""

    _collection_name = "node"
    __canonical_object_name__ = "Node"

    def first(self, **kwargs: Any) -> NoSQLNode:
        result = super().find_one(kwargs)
        if not result:
            raise NodeNotFoundError
        return result

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
        node = self.first(node_uid=node_uid)
        if node:
            attributes: Dict[str, Any] = {}
            inputs = {
                "node_name": node_name,
                "host_or_ip": host_or_ip,  # Consult with @Ionesio fields to update before merge.
                "is_vpn": is_vpn,
                "vpn_endpoint": vpn_endpoint,
                "vpn_key": vpn_key,
            }
            # TODO: refactor to update attr searchable in efficient way.
            for k, v in inputs.items():
                if k not in node.__attr_state__:
                    raise ValueError(f"Cannot set an non existing field:{k} to Node")
                else:
                    setattr(node, k, v)
                if k in node.__attr_searchable__:
                    attributes[k] = v

            attributes["__blob__"] = node.to_bytes()

            self.update_one(
                query={"node_uid": node_uid},
                values=attributes,
            )
        else:
            curr_len = len(self)
            node_row = NoSQLNode(
                id_int=curr_len + 1,
                node_uid=node_uid,
                node_name=node_name,
                host_or_ip=host_or_ip,
                is_vpn=is_vpn,
                vpn_endpoint=vpn_endpoint,
                vpn_key=vpn_key,
            )
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
        self, node_row: NoSQLNode, route_update: RouteUpdate, verify_key: VerifyKey
    ) -> bool:
        if not route_update.source_node_url:
            raise Exception("source_node_url is missing")
        source_url = GridURL.from_url(route_update.source_node_url)

        try:
            node = self.first(host_or_ip=source_url.host_or_ip, port=source_url.port)
        except NodeNotFoundError:
            node = None
        if node is None:
            # not created yet so valid
            return True
        elif node and node.node_uid == node_row.node_uid:
            # matches so valid
            return True
        # someone else already has this host and port so return False
        return False

    def update_route(
        self, node_row: NoSQLNode, route_update: RouteUpdate, is_vpn: bool = False
    ) -> None:
        if not route_update.source_node_url:
            raise Exception("source_node_url is missing")
        source_url = GridURL.from_url(route_update.source_node_url)

        try:
            node = self.first(
                host_or_ip=source_url.host_or_ip, node_uid=node_row.node_uid
            )
        except NodeNotFoundError:
            node = None
        if node:
            values = {
                "is_vpn": is_vpn,
                "private": route_update.private,
                "protocol": source_url.protocol,
                "port": source_url.port,
            }
            attributes = {}
            for k, v in values.items():
                if k not in node.__attr_state__:
                    raise ValueError(f"Cannot set an non existing field:{k} to Node")
                else:
                    setattr(node, k, v)
                if k in node.__attr_searchable__:
                    attributes[k] = v
            attributes["__blob__"] = node.to_bytes()

            self.update_one(
                query={
                    "host_or_ip": source_url.host_or_ip,
                    "node_uid": node_row.node_uid,
                },
                values=attributes,
            )
        else:
            raise ValueError(
                f"Route Update does not have an existing node: {node} to update."
            )

    # def get_routes(self, node_row: NoSQLNode) -> List[Dict]:
    #     # route_attrs = ["host_or_ip", "protocol", "port", "is_vpn", "private"]
    #     return self.query(node_uid=node_row.node_uid)
