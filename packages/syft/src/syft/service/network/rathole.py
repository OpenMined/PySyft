# stdlib
from typing import Self

# relative
from ...serde.serializable import serializable
from ...types.base import SyftBaseModel
from ...util.util import get_env
from .node_peer import NodePeer


def get_rathole_port() -> int:
    return int(get_env("RATHOLE_PORT", "2333"))


@serializable()
class RatholeConfig(SyftBaseModel):
    uuid: str
    secret_token: str
    local_addr_host: str
    local_addr_port: int
    server_name: str | None = None

    @property
    def local_address(self) -> str:
        return f"http://{self.local_addr_host}:{self.local_addr_port}"

    @classmethod
    def from_peer(cls, peer: NodePeer) -> Self:
        # relative
        from .routes import HTTPNodeRoute

        high_priority_route = peer.pick_highest_priority_route()

        if not isinstance(high_priority_route, HTTPNodeRoute):
            raise ValueError("Rathole only supports HTTPNodeRoute")

        return cls(
            uuid=peer.id,
            secret_token=peer.rathole_token,
            local_addr_host=high_priority_route.host_or_ip,
            local_addr_port=high_priority_route.port,
            server_name=peer.name,
        )
