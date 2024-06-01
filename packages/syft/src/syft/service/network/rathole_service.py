# stdlib
import secrets
from typing import cast

# third party
from kr8s.objects import Service
import yaml

# relative
from ...custom_worker.k8s import KubeUtils
from ...custom_worker.k8s import get_kr8s_client
from ...types.uid import UID
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

        rathole_route = peer.get_rathole_route()
        if not rathole_route:
            raise Exception(f"Peer: {peer} has no rathole route: {rathole_route}")

        random_port = self.get_random_port()

        peer_id = cast(UID, peer.id)

        config = RatholeConfig(
            uuid=peer_id.to_string(),
            secret_token=rathole_route.rathole_token,
            local_addr_host="0.0.0.0",
            local_addr_port=random_port,
            server_name=peer.name,
        )

        # Get rathole toml config map
        rathole_config_map = KubeUtils.get_configmap(
            client=self.k8rs_client, name=RATHOLE_TOML_CONFIG_MAP
        )

        if rathole_config_map is None:
            raise Exception("Rathole config map not found.")

        client_filename = RatholeServerToml.filename

        toml_str = rathole_config_map.data[client_filename]

        # Add the peer info to the toml file
        rathole_toml = RatholeServerToml(toml_str)
        rathole_toml.add_config(config=config)

        # First time adding a peer
        if not rathole_toml.get_rathole_listener_addr():
            bind_addr = f"localhost:{get_rathole_port()}"
            rathole_toml.set_rathole_listener_addr(bind_addr)

        data = {client_filename: rathole_toml.toml_str}

        # Update the rathole config map
        KubeUtils.update_configmap(config_map=rathole_config_map, patch={"data": data})

        # Add the peer info to the proxy config map
        self.add_dynamic_addr_to_rathole(config)

    def get_random_port(self) -> int:
        """Get a random port number."""
        return secrets.randbits(15)

    def add_host_to_client(
        self, peer_name: str, peer_id: str, rathole_token: str, remote_addr: str
    ) -> None:
        """Add a host to the rathole client toml file."""

        config = RatholeConfig(
            uuid=peer_id,
            secret_token=rathole_token,
            local_addr_host="proxy",
            local_addr_port=80,
            server_name=peer_name,
        )

        # Get rathole toml config map
        rathole_config_map = KubeUtils.get_configmap(
            client=self.k8rs_client, name=RATHOLE_TOML_CONFIG_MAP
        )

        if rathole_config_map is None:
            raise Exception("Rathole config map not found.")

        client_filename = RatholeClientToml.filename

        toml_str = rathole_config_map.data[client_filename]

        rathole_toml = RatholeClientToml(toml_str=toml_str)

        rathole_toml.add_config(config=config)

        rathole_toml.set_remote_addr(remote_addr)

        data = {client_filename: rathole_toml.toml_str}

        # Update the rathole config map
        KubeUtils.update_configmap(config_map=rathole_config_map, patch={"data": data})

    def add_dynamic_addr_to_rathole(
        self, config: RatholeConfig, entrypoint: str = "web"
    ) -> None:
        """Add a port to the rathole proxy config map."""

        rathole_proxy_config_map = KubeUtils.get_configmap(
            self.k8rs_client, RATHOLE_PROXY_CONFIG_MAP
        )

        if rathole_proxy_config_map is None:
            raise Exception("Rathole proxy config map not found.")

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

        proxy_rule = (
            f"Host(`{config.server_name}.syft.local`) || "
            f"HostHeader(`{config.server_name}.syft.local`) && PathPrefix(`/`)"
        )

        rathole_proxy["http"]["routers"][config.server_name] = {
            "rule": proxy_rule,
            "service": config.server_name,
            "entryPoints": [entrypoint],
        }

        KubeUtils.update_configmap(
            config_map=rathole_proxy_config_map,
            patch={"data": {"rathole-dynamic.yml": yaml.safe_dump(rathole_proxy)}},
        )

        self.expose_port_on_rathole_service(config.server_name, config.local_addr_port)

    def expose_port_on_rathole_service(self, port_name: str, port: int) -> None:
        """Expose a port on the rathole service."""

        rathole_service = KubeUtils.get_service(self.k8rs_client, "rathole")

        rathole_service = cast(Service, rathole_service)

        config = rathole_service.raw

        existing_port_idx = None
        for idx, existing_port in enumerate(config["spec"]["ports"]):
            if existing_port["name"] == port_name:
                print("Port already exists.", existing_port_idx, port_name)
                existing_port_idx = idx
                break

        if existing_port_idx is not None:
            config["spec"]["ports"][existing_port_idx]["port"] = port
            config["spec"]["ports"][existing_port_idx]["targetPort"] = port
        else:
            config["spec"]["ports"].append(
                {
                    "name": port_name,
                    "port": port,
                    "targetPort": port,
                    "protocol": "TCP",
                }
            )

        rathole_service.patch(config)

    def remove_port_on_rathole_service(self, port_name: str) -> None:
        """Remove a port from the rathole service."""

        rathole_service = KubeUtils.get_service(self.k8rs_client, "rathole")

        rathole_service = cast(Service, rathole_service)

        config = rathole_service.raw

        ports = config["spec"]["ports"]

        for port in ports:
            if port["name"] == port_name:
                ports.remove(port)
                break

        rathole_service.patch(config)
