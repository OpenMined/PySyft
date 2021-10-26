# third party
from sqlalchemy.engine import Engine

# relative
from ..node_table.node_route import NodeRoute
from .database_manager import DatabaseManager


class NodeRouteManager(DatabaseManager):
    schema = NodeRoute

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=NodeRouteManager.schema, db=database)

    def update_route_for_node(
        self, node_id: int, host_or_ip: str, is_vpn: bool = False
    ) -> NodeRoute:
        # node_id is a database int id
        # host_or_ip can have a port as well

        node_route = self.first(host_or_ip=host_or_ip)
        if node_route:
            self.modify(
                query={"host_or_ip": host_or_ip},
                values={"is_vpn": is_vpn, "node_id": node_id},
            )
        else:
            self.register(
                **{"node_id": node_id, "host_or_ip": host_or_ip, "is_vpn": is_vpn}
            )
            node_route = self.first(host_or_ip=host_or_ip)
        return node_route
