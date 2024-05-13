# stdlib
import secrets

# third party
import yaml

# relative
from ...custom_worker.k8s import KubeUtils
from ...custom_worker.k8s import get_kr8s_client
from .node_peer import NodePeer
from .rathole import RatholeConfig
from .rathole import get_rathole_port
from .rathole_toml import RatholeClientToml
from .rathole_toml import RatholeServerToml

RATHOLE_TOML_CONFIG_MAP = "rathole-config"
RATHOLE_PROXY_CONFIG_MAP = "proxy-config-dynamic"
PROXY_CONFIG_MAP = "proxy-config"


class RatholeService:
    def __init__(self) -> None:
        self.k8rs_client = get_kr8s_client()

    def add_host_to_server(self, peer: NodePeer) -> None:
        """Add a host to the rathole server toml file.

        Args:
            peer (NodePeer): The peer to be added to the rathole server.

        Returns:
            None
        """

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

        client_filename = RatholeServerToml.filename

        toml_str = rathole_config_map.data[client_filename]

        # Add the peer info to the toml file
        rathole_toml = RatholeServerToml(toml_str)
        rathole_toml.add_config(config=config)

        # First time adding a peer
        if not rathole_toml.get_rathole_listener_addr():
            bind_addr = f"http://localhost:{get_rathole_port()}"
            rathole_toml.set_rathole_listener_addr(bind_addr)

        data = {client_filename: rathole_toml.toml_str}

        # Update the rathole config map
        KubeUtils.update_configmap(config_map=rathole_config_map, patch={"data": data})

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

        data = {client_filename: rathole_toml.toml_str}

        # Update the rathole config map
        KubeUtils.update_configmap(config_map=rathole_config_map, patch={"data": data})

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
        else:
            rathole_proxy = yaml.safe_load(rathole_proxy)

        rathole_proxy["http"]["services"][config.server_name] = {
            "loadBalancer": {"servers": [{"url": "http://proxy:8001"}]}
        }

        rathole_proxy["http"]["routers"][config.server_name] = {
            "rule": "PathPrefix(`/`)",
            "service": config.server_name,
            "entryPoints": [entrypoint],
        }

        KubeUtils.update_configmap(
            config_map=rathole_proxy_config_map,
            patch={"data": {"rathole-dynamic.yml": yaml.safe_dump(rathole_proxy)}},
        )

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
        else:
            rathole_proxy = yaml.safe_load(rathole_proxy)

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

        KubeUtils.update_configmap(
            config_map=rathole_proxy_config_map,
            patch={"data": {"rathole-dynamic.yml": yaml.safe_dump(rathole_proxy)}},
        )

    def add_entrypoint(self, port: int, peer_name: str) -> None:
        """Add an entrypoint to the traefik config map."""

        proxy_config_map = KubeUtils.get_configmap(self.k8rs_client, PROXY_CONFIG_MAP)

        data = proxy_config_map.data

        traefik_config_str = data["traefik.yml"]

        traefik_config = yaml.safe_load(traefik_config_str)

        traefik_config["entryPoints"][f"{peer_name}"] = {"address": f":{port}"}

        data["traefik.yml"] = yaml.safe_dump(traefik_config)

        KubeUtils.update_configmap(config_map=proxy_config_map, patch={"data": data})

    def remove_endpoint(self, peer_name: str) -> None:
        """Remove an entrypoint from the traefik config map."""

        proxy_config_map = KubeUtils.get_configmap(self.k8rs_client, PROXY_CONFIG_MAP)

        data = proxy_config_map.data

        traefik_config_str = data["traefik.yml"]

        traefik_config = yaml.safe_load(traefik_config_str)

        del traefik_config["entryPoints"][f"{peer_name}"]

        data["traefik.yml"] = yaml.safe_dump(traefik_config)

        KubeUtils.update_configmap(config_map=proxy_config_map, patch={"data": data})
