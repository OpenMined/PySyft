# stdlib
from typing import List

# third party
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine

# relative
from .....grid import GridURL
from ..node_service.node_route.route_update import RouteUpdate
from ..node_table.node import Node as NodeRow
from ..node_table.node_route import NodeRoute
from .database_manager import DatabaseManager


class NodeRouteManager(DatabaseManager):
    schema = NodeRoute

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=NodeRouteManager.schema, db=database)

    def update_route_for_node(
        self,
        node_id: int,
        host_or_ip: str,
        is_vpn: bool = False,
        vpn_endpoint: str = "",
        vpn_key: str = "",
    ) -> NodeRoute:
        # node_id is a database int id
        # host_or_ip can have a port as well

        node_route = self.first(host_or_ip=host_or_ip)
        values = {"is_vpn": is_vpn, "node_id": node_id, "host_or_ip": host_or_ip}

        # Only change optional columns if parameters aren't empty strings.
        if vpn_endpoint:
            values["vpn_endpoint"] = vpn_endpoint
        if vpn_key:
            values["vpn_key"] = vpn_key

        if node_route:
            self.modify(
                query={"host_or_ip": host_or_ip},
                values=values,
            )
        else:
            values["node_id"] = node_id
            self.register(**values)
            node_route = self.first(host_or_ip=host_or_ip)
        return node_route

    def validate_route_update(
        self, node_row: NodeRow, route_update: RouteUpdate, verify_key: VerifyKey
    ) -> bool:
        if not route_update.source_node_url:
            raise Exception("source_node_url is missing")
        source_url = GridURL.from_url(route_update.source_node_url)

        node_route = self.first(host_or_ip=source_url.host_or_ip, port=source_url.port)
        if node_route is None:
            # not created yet so valid
            return True
        elif node_route and node_route.node_id == node_row.id:
            # matches so valid
            return True
        # someone else already has this host and port so return False
        return False

    def update_route(
        self, node_row: NodeRow, route_update: RouteUpdate, is_vpn: bool = False
    ) -> NodeRoute:
        if not route_update.source_node_url:
            raise Exception("source_node_url is missing")
        source_url = GridURL.from_url(route_update.source_node_url)

        node_route = self.first(host_or_ip=source_url.host_or_ip, node_id=node_row.id)
        if node_route:
            self.modify(
                query={"host_or_ip": source_url.host_or_ip, "node_id": node_row.id},
                values={
                    "is_vpn": is_vpn,
                    "private": route_update.private,
                    "protocol": source_url.protocol,
                    "port": source_url.port,
                },
            )
        else:
            self.register(
                **{
                    "is_vpn": is_vpn,
                    "node_id": node_row.id,
                    "private": route_update.private,
                    "protocol": source_url.protocol,
                    "port": source_url.port,
                    "host_or_ip": source_url.host_or_ip,
                }
            )
        new_node_route_row = self.first(
            host_or_ip=source_url.host_or_ip, node_id=node_row.id
        )
        if new_node_route_row:
            return new_node_route_row
        raise Exception("Failed to update node_route")

    def get_routes(self, node_row: NodeRow) -> List[NodeRoute]:
        return self.query(node_id=node_row.id)
