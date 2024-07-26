# relative
from ...types.server_url import ServerURL
from .rathole_config_builder import RatholeConfigBuilder
from .routes import ServerRoute
from .server_peer import ServerPeer


class ReverseTunnelService:
    def __init__(self) -> None:
        self.builder = RatholeConfigBuilder()

    def set_client_config(
        self,
        self_server_peer: ServerPeer,
        remote_server_route: ServerRoute,
    ) -> None:
        rathole_route = self_server_peer.get_rtunnel_route()
        if not rathole_route:
            raise Exception(
                "Failed to exchange routes via . "
                + f"Peer: {self_server_peer} has no rathole route: {rathole_route}"
            )

        remote_url = ServerURL(
            host_or_ip=remote_server_route.host_or_ip, port=remote_server_route.port
        )
        rathole_remote_addr = remote_url.as_container_host()

        remote_addr = rathole_remote_addr.url_no_protocol

        self.builder.add_host_to_client(
            peer_name=self_server_peer.name,
            peer_id=str(self_server_peer.id),
            rtunnel_token=rathole_route.rtunnel_token,
            remote_addr=remote_addr,
        )

    def set_server_config(self, remote_peer: ServerPeer) -> None:
        rathole_route = remote_peer.get_rtunnel_route()
        self.builder.add_host_to_server(remote_peer) if rathole_route else None

    def clear_client_config(self, self_server_peer: ServerPeer) -> None:
        self.builder.remove_host_from_client(str(self_server_peer.id))

    def clear_server_config(self, remote_peer: ServerPeer) -> None:
        self.builder.remove_host_from_server(
            str(remote_peer.id), server_name=remote_peer.name
        )
