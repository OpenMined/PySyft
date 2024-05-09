# stdlib
import secrets
from typing import Self
from typing import cast

# third party
import yaml

# relative
from ...custom_worker.k8s import KubeUtils
from ...custom_worker.k8s import get_kr8s_client
from ...serde import serializable
from ...types.base import SyftBaseModel
from .node_peer import NodePeer
from .rathole_toml import RatholeClientToml
from .rathole_toml import RatholeServerToml
from .routes import HTTPNodeRoute

RATHOLE_TOML_CONFIG_MAP = "rathole-config"
RATHOLE_PROXY_CONFIG_MAP = "rathole-proxy-config"
RATHOLE_DEFAULT_BIND_ADDRESS = "http://0.0.0.0:2333"
PROXY_CONFIG_MAP = "proxy-config"


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


# class RatholeProxyConfigWriter:
#     def get_config(self, *args, **kwargs):
#         pass

#     def save_config(self, *args, **kwargs):
#         pass

#     def add_service(url: str, service_name: str, port: int, hostname: str):
#         pass

#     def delete_service(self, *args, **kwargs):
#         pass


class RatholeService:
    def __init__(self) -> None:
        self.k8rs_client = get_kr8s_client()

    def add_host_to_server(self, peer: NodePeer) -> None:
        """Add a host to the rathole server toml file."""

        route = cast(HTTPNodeRoute, peer.pick_highest_priority_route())

        config = RatholeConfig(
            uuid=peer.id.to_string(),
            secret_token=peer.rathole_token,
            local_addr_host="localhost",
            local_addr_port=route.port,
            server_name=peer.name,
        )

        # Get rathole toml config map
        rathole_config_map = KubeUtils.get_configmap(
            client=self.k8rs_client, name=RATHOLE_TOML_CONFIG_MAP
        )

        client_filename = RatholeServerToml.filename

        toml_str = rathole_config_map.data[client_filename]

        # Add the peer info to the toml file
        rathole_toml = RatholeServerToml(toml_str)
        rathole_toml.add_config(config=config)

        if not rathole_toml.get_bind_address():
            # First time adding a peer
            rathole_toml.set_bind_address(RATHOLE_DEFAULT_BIND_ADDRESS)

        rathole_config_map.data[client_filename] = rathole_toml.toml_str

        # Update the rathole config map
        KubeUtils.update_configmap(
            client=self.k8rs_client,
            name=RATHOLE_TOML_CONFIG_MAP,
            data=rathole_config_map.data,
        )

        # Add the peer info to the proxy config map
        self.add_dynamic_addr_to_rathole(config)

    def get_random_port(self) -> int:
        """Get a random port number."""
        return secrets.randbits(15)

    def add_host_to_client(self, peer: NodePeer) -> None:
        """Add a host to the rathole client toml file."""

        random_port = self.get_random_port()

        config = RatholeConfig(
            uuid=peer.id.to_string(),
            secret_token=peer.rathole_token,
            local_addr_host="localhost",
            local_addr_port=random_port,
            server_name=peer.name,
        )

        # Get rathole toml config map
        rathole_config_map = KubeUtils.get_configmap(
            client=self.k8rs_client, name=RATHOLE_TOML_CONFIG_MAP
        )

        client_filename = RatholeClientToml.filename

        toml_str = rathole_config_map.data[client_filename]

        rathole_toml = RatholeClientToml(toml_str=toml_str)

        rathole_toml.add_config(config=config)

        self.add_entrypoint(port=random_port, peer_name=peer.name)

        self.forward_port_to_proxy(config=config, entrypoint=peer.name)

    def forward_port_to_proxy(
        self, config: RatholeConfig, entrypoint: str = "web"
    ) -> None:
        """Add a port to the rathole proxy config map."""

        rathole_proxy_config_map = KubeUtils.get_configmap(
            self.k8rs_client, RATHOLE_PROXY_CONFIG_MAP
        )

        rathole_proxy = rathole_proxy_config_map.data["rathole-dynamic.yml"]

        if not rathole_proxy:
            rathole_proxy = {"http": {"routers": {}, "services": {}}}

        rathole_proxy["http"]["services"][config.server_name] = {
            "loadBalancer": {"servers": [{"url": "http://proxy:8001"}]}
        }

        rathole_proxy["http"]["routers"][config.server_name] = {
            "rule": "PathPrefix(`/`)",
            "service": config.server_name,
            "entryPoints": [entrypoint],
        }

        KubeUtils.update_configmap(self.k8rs_client, PROXY_CONFIG_MAP, rathole_proxy)

    def add_dynamic_addr_to_rathole(
        self, config: RatholeConfig, entrypoint: str = "web"
    ) -> None:
        """Add a port to the rathole proxy config map."""

        rathole_proxy_config_map = KubeUtils.get_configmap(
            self.k8rs_client, RATHOLE_PROXY_CONFIG_MAP
        )

        rathole_proxy = rathole_proxy_config_map.data["rathole-dynamic.yml"]

        if not rathole_proxy:
            rathole_proxy = {"http": {"routers": {}, "services": {}}}

        rathole_proxy["http"]["services"][config.server_name] = {
            "loadBalancer": {
                "servers": [{"url": f"http://rathole:{config.local_addr_port}"}]
            }
        }

        rathole_proxy["http"]["routers"][config.server_name] = {
            "rule": f"Host(`{config.server_name}.syft.local`)",
            "service": config.server_name,
            "entryPoints": [entrypoint],
        }

        KubeUtils.update_configmap(self.k8rs_client, PROXY_CONFIG_MAP, rathole_proxy)

    def add_entrypoint(self, port: int, peer_name: str) -> None:
        """Add an entrypoint to the traefik config map."""

        proxy_config_map = KubeUtils.get_configmap(self.k8rs_client, PROXY_CONFIG_MAP)

        data = proxy_config_map.data

        traefik_config_str = data["traefik.yml"]

        traefik_config = yaml.safe_load(traefik_config_str)

        traefik_config["entryPoints"][f"{peer_name}"] = {"address": f":{port}"}

        data["traefik.yml"] = yaml.safe_dump(traefik_config)

        KubeUtils.update_configmap(self.k8rs_client, PROXY_CONFIG_MAP, data)

    def remove_endpoint(self, peer_name: str) -> None:
        """Remove an entrypoint from the traefik config map."""

        proxy_config_map = KubeUtils.get_configmap(self.k8rs_client, PROXY_CONFIG_MAP)

        data = proxy_config_map.data

        traefik_config_str = data["traefik.yml"]

        traefik_config = yaml.safe_load(traefik_config_str)

        del traefik_config["entryPoints"][f"{peer_name}"]

        data["traefik.yml"] = yaml.safe_dump(traefik_config)

        KubeUtils.update_configmap(self.k8rs_client, PROXY_CONFIG_MAP, data)
