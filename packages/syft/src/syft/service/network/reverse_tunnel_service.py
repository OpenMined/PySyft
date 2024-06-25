# relative
from ...types.grid_url import GridURL
from .node_peer import NodePeer
from .rathole_config_builder import RatholeConfigBuilder
from .routes import NodeRoute


class ReverseTunnelService:
    def __init__(self) -> None:
        self.builder = RatholeConfigBuilder()

    def set_client_config(
        self,
        self_node_peer: NodePeer,
        remote_node_route: NodeRoute,
    ) -> None:
        rathole_route = self_node_peer.get_rathole_route()
        if not rathole_route:
            raise Exception(
                "Failed to exchange routes via . "
                + f"Peer: {self_node_peer} has no rathole route: {rathole_route}"
            )

        remote_url = GridURL(
            host_or_ip=remote_node_route.host_or_ip, port=remote_node_route.port
        )
        rathole_remote_addr = remote_url.as_container_host()

        remote_addr = rathole_remote_addr.url_no_protocol

        self.builder.add_host_to_client(
            peer_name=self_node_peer.name,
            peer_id=str(self_node_peer.id),
            rathole_token=rathole_route.rathole_token,
            remote_addr=remote_addr,
        )

    def set_server_config(self, remote_peer: NodePeer) -> None:
        rathole_route = remote_peer.get_rathole_route()
        self.builder.add_host_to_server(remote_peer) if rathole_route else None

    def clear_client_config(self, self_node_peer: NodePeer) -> None:
        self.builder.remove_host_from_client(str(self_node_peer.id))

    def clear_server_config(self, remote_peer: NodePeer) -> None:
        self.builder.remove_host_from_server(
            str(remote_peer.id), server_name=remote_peer.name
        )
