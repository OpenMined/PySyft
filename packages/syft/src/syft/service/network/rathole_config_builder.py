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
from .rathole import RatholeConfig
from .rathole import get_rathole_port
from .rathole_toml import RatholeClientToml
from .rathole_toml import RatholeServerToml
from .server_peer import ServerPeer

RATHOLE_TOML_CONFIG_MAP = "rathole-config"
RATHOLE_PROXY_CONFIG_MAP = "proxy-config-dynamic"
PROXY_CONFIG_MAP = "proxy-config"
DEFAULT_LOCAL_ADDR_HOST = "0.0.0.0"  # nosec


class RatholeConfigBuilder:
    def __init__(self) -> None:
        self.k8rs_client = get_kr8s_client()

    def add_host_to_server(self, peer: ServerPeer) -> None:
        """Add a host to the rathole server toml file.

        Args:
            peer (ServerPeer): The peer to be added to the rathole server.

        Returns:
            None
        """

        rathole_route = peer.get_rtunnel_route()
        if not rathole_route:
            raise Exception(f"Peer: {peer} has no rathole route: {rathole_route}")

        random_port = self._get_random_port()

        peer_id = cast(UID, peer.id)

        config = RatholeConfig(
            uuid=peer_id.to_string(),
            secret_token=rathole_route.rtunnel_token,
            local_addr_host=DEFAULT_LOCAL_ADDR_HOST,
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
        self._add_dynamic_addr_to_rathole(config)

    def remove_host_from_server(self, peer_id: str, server_name: str) -> None:
        """Remove a host from the rathole server toml file.

        Args:
            peer_id (str): The id of the peer to be removed.
            server_name (str): The name of the peer to be removed.

        Returns:
            None
        """

        rathole_config_map = KubeUtils.get_configmap(
            client=self.k8rs_client, name=RATHOLE_TOML_CONFIG_MAP
        )

        if rathole_config_map is None:
            raise Exception("Rathole config map not found.")

        client_filename = RatholeServerToml.filename

        toml_str = rathole_config_map.data[client_filename]

        rathole_toml = RatholeServerToml(toml_str=toml_str)

        rathole_toml.remove_config(peer_id)

        data = {client_filename: rathole_toml.toml_str}

        # Update the rathole config map
        KubeUtils.update_configmap(config_map=rathole_config_map, patch={"data": data})

        # Remove the peer info from the proxy config map
        self._remove_dynamic_addr_from_rathole(server_name)

    def _get_random_port(self) -> int:
        """Get a random port number."""
        return secrets.randbits(15)

    def add_host_to_client(
        self, peer_name: str, peer_id: str, rtunnel_token: str, remote_addr: str
    ) -> None:
        """Add a host to the rathole client toml file."""

        config = RatholeConfig(
            uuid=peer_id,
            secret_token=rtunnel_token,
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

    def remove_host_from_client(self, peer_id: str) -> None:
        """Remove a host from the rathole client toml file."""

        rathole_config_map = KubeUtils.get_configmap(
            client=self.k8rs_client, name=RATHOLE_TOML_CONFIG_MAP
        )

        if rathole_config_map is None:
            raise Exception("Rathole config map not found.")

        client_filename = RatholeClientToml.filename

        toml_str = rathole_config_map.data[client_filename]

        rathole_toml = RatholeClientToml(toml_str=toml_str)

        rathole_toml.remove_config(peer_id)

        rathole_toml.clear_remote_addr()

        data = {client_filename: rathole_toml.toml_str}

        # Update the rathole config map
        KubeUtils.update_configmap(config_map=rathole_config_map, patch={"data": data})

    def _add_dynamic_addr_to_rathole(
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
            rathole_proxy = {"http": {"routers": {}, "services": {}, "middlewares": {}}}
        else:
            rathole_proxy = yaml.safe_load(rathole_proxy)

        rathole_proxy["http"]["services"][config.server_name] = {
            "loadBalancer": {
                "servers": [{"url": f"http://rathole:{config.local_addr_port}"}]
            }
        }

        rathole_proxy["http"]["middlewares"]["strip-rathole-prefix"] = {
            "replacePathRegex": {"regex": "^/rathole/(.*)", "replacement": "/$1"}
        }

        proxy_rule = (
            f"Host(`{config.server_name}.syft.local`) || "
            f"HostHeader(`{config.server_name}.syft.local`) && PathPrefix(`/rtunnel`)"
        )

        rathole_proxy["http"]["routers"][config.server_name] = {
            "rule": proxy_rule,
            "service": config.server_name,
            "entryPoints": [entrypoint],
            "middlewares": ["strip-rathole-prefix"],
        }

        KubeUtils.update_configmap(
            config_map=rathole_proxy_config_map,
            patch={"data": {"rathole-dynamic.yml": yaml.safe_dump(rathole_proxy)}},
        )

        self._expose_port_on_rathole_service(config.server_name, config.local_addr_port)

    def _remove_dynamic_addr_from_rathole(self, server_name: str) -> None:
        """Remove a port from the rathole proxy config map."""

        rathole_proxy_config_map = KubeUtils.get_configmap(
            self.k8rs_client, RATHOLE_PROXY_CONFIG_MAP
        )

        if rathole_proxy_config_map is None:
            raise Exception("Rathole proxy config map not found.")

        rathole_proxy = rathole_proxy_config_map.data["rathole-dynamic.yml"]

        if not rathole_proxy:
            return

        rathole_proxy = yaml.safe_load(rathole_proxy)

        if server_name in rathole_proxy["http"]["routers"]:
            del rathole_proxy["http"]["routers"][server_name]

        if server_name in rathole_proxy["http"]["services"]:
            del rathole_proxy["http"]["services"][server_name]

        KubeUtils.update_configmap(
            config_map=rathole_proxy_config_map,
            patch={"data": {"rathole-dynamic.yml": yaml.safe_dump(rathole_proxy)}},
        )

        self._remove_port_on_rathole_service(server_name)

    def _expose_port_on_rathole_service(self, port_name: str, port: int) -> None:
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

    def _remove_port_on_rathole_service(self, port_name: str) -> None:
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
