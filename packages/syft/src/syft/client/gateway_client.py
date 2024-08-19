# stdlib
from typing import Any

# relative
from ..abstract_server import ServerSideType
from ..abstract_server import ServerType
from ..serde.serializable import serializable
from ..server.credentials import SyftSigningKey
from ..service.metadata.server_metadata import ServerMetadataJSON
from ..service.network.server_peer import ServerPeer
from ..types.errors import SyftException
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..util.assets import load_png_base64
from ..util.notebook_ui.styles import FONT_CSS
from .client import SyftClient
from .connection import ServerConnection


@serializable(canonical_name="GatewayClient", version=1)
class GatewayClient(SyftClient):
    # TODO: add widget repr for gateway client

    def proxy_to(self, peer: Any) -> SyftClient:
        # relative
        from .datasite_client import DatasiteClient
        from .enclave_client import EnclaveClient

        connection: type[ServerConnection] = self.connection.with_proxy(peer.id)
        metadata: ServerMetadataJSON = connection.get_server_metadata(
            credentials=SyftSigningKey.generate()
        )
        if metadata.server_type == ServerType.DATASITE.value:
            client_type: type[SyftClient] = DatasiteClient
        elif metadata.server_type == ServerType.ENCLAVE.value:
            client_type = EnclaveClient
        else:
            raise SyftException(
                public_message=f"Unknown server type {metadata.server_type} to create proxy client"
            )

        client = client_type(
            connection=connection,
            credentials=self.credentials,
        )
        return client

    def proxy_client_for(
        self,
        name: str,
        email: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> SyftClient:
        peer = None
        if self.api.has_service("network"):
            peer = self.api.services.network.get_peer_by_name(name=name)
        if peer is None:
            raise SyftException(public_message=f"No datasite with name {name}")
        res = self.proxy_to(peer)
        if email and password:
            res = res.login(email=email, password=password, **kwargs)
        return res

    @property
    def peers(self) -> list[ServerPeer] | None:
        return ProxyClient(routing_client=self)

    @property
    def datasites(self) -> list[ServerPeer] | None:
        return ProxyClient(routing_client=self, server_type=ServerType.DATASITE)

    @property
    def enclaves(self) -> list[ServerPeer] | None:
        return ProxyClient(routing_client=self, server_type=ServerType.ENCLAVE)

    def _repr_html_(self) -> str:
        commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;
        .datasites</span> - list datasites connected to this gateway</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;
        .proxy_client_for</span> - get a connection to a listed datasite</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;
        .login</span> - log into the gateway</li>
        """

        command_list = f"""
        <ul style='padding-left: 1em;'>
            {commands}
        </ul>
        """

        small_server_symbol_logo = load_png_base64("small-syft-symbol-logo.png")

        url = getattr(self.connection, "url", None)
        server_details = f"<strong>URL:</strong> {url}<br />" if url else ""
        if self.metadata:
            server_details += f"<strong>Server Type:</strong> {self.metadata.server_type.capitalize()}<br />"
            server_side_type = (
                "Low Side"
                if self.metadata.server_side_type == ServerSideType.LOW_SIDE.value
                else "High Side"
            )
            server_details += (
                f"<strong>Server Side Type:</strong> {server_side_type}<br />"
            )
            server_details += (
                f"<strong>Syft Version:</strong> {self.metadata.syft_version}<br />"
            )

        return f"""
        <style>
            {FONT_CSS}

            .syft-container {{
                padding: 5px;
                font-family: 'Open Sans';
            }}
            .syft-alert-info {{
                color: #1F567A;
                background-color: #C2DEF0;
                border-radius: 4px;
                padding: 5px;
                padding: 13px 10px
            }}
            .syft-code-block {{
                background-color: #f7f7f7;
                border: 1px solid #cfcfcf;
                padding: 0px 2px;
            }}
            .syft-space {{
                margin-top: 1em;
            }}
        </style>
        <div class="syft-client syft-container">
            <img src="{small_server_symbol_logo}" alt="Logo"
            style="width:48px;height:48px;padding:3px;">
            <h2>Welcome to {self.name}</h2>
            <div class="syft-space">
                {server_details}
            </div>
            <div class='syft-alert-info syft-space'>
                &#9432;&nbsp;
                This server is run by the library PySyft to learn more about how it works visit
                <a href="https://github.com/OpenMined/PySyft">github.com/OpenMined/PySyft</a>.
            </div>
            <h4>Commands to Get Started</h4>
            {command_list}
        </div><br />
        """


class ProxyClient(SyftObject):
    __canonical_name__ = "ProxyClient"
    __version__ = SYFT_OBJECT_VERSION_1

    routing_client: GatewayClient
    server_type: ServerType | None = None

    def retrieve_servers(self) -> list[ServerPeer]:
        if self.server_type in [ServerType.DATASITE, ServerType.ENCLAVE]:
            return self.routing_client.api.services.network.get_peers_by_type(
                server_type=self.server_type
            )
        elif self.server_type is None:
            # if server type is None, return all servers
            return self.routing_client.api.services.network.get_all_peers()
        else:
            raise SyftException(
                public_message=f"Unknown server type {self.server_type} to retrieve proxy client"
            )

    def _repr_html_(self) -> str:
        return self.retrieve_servers()._repr_html_()

    def __len__(self) -> int:
        return len(self.retrieve_servers())

    def __getitem__(self, key: int | str) -> SyftClient:
        if not isinstance(key, int):
            raise SyftException(public_message=f"Key: {key} must be an integer")

        servers = self.retrieve_servers()

        if key >= len(servers):
            raise SyftException(
                public_message=f"Index {key} out of range for retrieved servers"
            )

        return self.routing_client.proxy_to(servers[key])
