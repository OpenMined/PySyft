# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from typing_extensions import Self

# relative
from ..abstract_node import NodeSideType
from ..abstract_node import NodeType
from ..img.base64 import base64read
from ..node.credentials import SyftSigningKey
from ..serde.serializable import serializable
from ..service.network.node_peer import NodePeer
from ..service.response import SyftError
from ..service.response import SyftException
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..util.fonts import fonts_css
from .client import SyftClient


@serializable()
class GatewayClient(SyftClient):
    # TODO: add widget repr for gateway client

    def proxy_to(self, peer: Any) -> Self:
        # relative
        from .domain_client import DomainClient
        from .enclave_client import EnclaveClient

        connection = self.connection.with_proxy(peer.id)
        metadata = connection.get_node_metadata(credentials=SyftSigningKey.generate())
        if metadata.node_type == NodeType.DOMAIN.value:
            client_type = DomainClient
        elif metadata.node_type == NodeType.ENCLAVE.value:
            client_type = EnclaveClient
        else:
            raise SyftException(
                f"Unknown node type {metadata.node_type} to create proxy client"
            )

        client = client_type(
            connection=connection,
            credentials=self.credentials,
        )
        return client

    def proxy_client_for(
        self,
        name: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        peer = None
        if self.api.has_service("network"):
            peer = self.api.services.network.get_peer_by_name(name=name)
        if peer is None:
            return SyftError(message=f"No domain with name {name}")
        res = self.proxy_to(peer)
        if email and password:
            res = res.login(email=email, password=password, **kwargs)
        return res

    @property
    def peers(self) -> Optional[Union[List[NodePeer], SyftError]]:
        return ProxyClient(routing_client=self)

    @property
    def domains(self) -> Optional[Union[List[NodePeer], SyftError]]:
        return ProxyClient(routing_client=self, node_type=NodeType.DOMAIN)

    @property
    def enclaves(self) -> Optional[Union[List[NodePeer], SyftError]]:
        return ProxyClient(routing_client=self, node_type=NodeType.ENCLAVE)

    def _repr_html_(self) -> str:
        commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;
        .domains</span> - list domains connected to this gateway</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;
        .proxy_client_for</span> - get a connection to a listed domain</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;
        .login</span> - log into the gateway</li>
        """

        command_list = f"""
        <ul style='padding-left: 1em;'>
            {commands}
        </ul>
        """

        small_grid_symbol_logo = base64read("small-grid-symbol-logo.png")

        url = getattr(self.connection, "url", None)
        node_details = f"<strong>URL:</strong> {url}<br />" if url else ""
        node_details += (
            f"<strong>Node Type:</strong> {self.metadata.node_type.capitalize()}<br />"
        )
        node_side_type = (
            "Low Side"
            if self.metadata.node_side_type == NodeSideType.LOW_SIDE.value
            else "High Side"
        )
        node_details += f"<strong>Node Side Type:</strong> {node_side_type}<br />"
        node_details += (
            f"<strong>Syft Version:</strong> {self.metadata.syft_version}<br />"
        )

        return f"""
        <style>
            {fonts_css}

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
            <img src="{small_grid_symbol_logo}" alt="Logo"
            style="width:48px;height:48px;padding:3px;">
            <h2>Welcome to {self.name}</h2>
            <div class="syft-space">
                {node_details}
            </div>
            <div class='syft-alert-info syft-space'>
                &#9432;&nbsp;
                This node is run by the library PySyft to learn more about how it works visit
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
    node_type: Optional[NodeType]

    def retrieve_nodes(self) -> List[NodePeer]:
        if self.node_type in [NodeType.DOMAIN, NodeType.ENCLAVE]:
            return self.routing_client.api.services.network.get_peers_by_type(
                node_type=self.node_type
            )
        elif self.node_type is None:
            # if node type is None, return all nodes
            return self.routing_client.api.services.network.get_all_peers()
        else:
            raise SyftException(
                f"Unknown node type {self.node_type} to retrieve proxy client"
            )

    def _repr_html_(self) -> str:
        return self.retrieve_nodes()._repr_html_()

    def __len__(self) -> int:
        return len(self.retrieve_nodes())

    def __getitem__(self, key: int):
        if not isinstance(key, int):
            raise SyftException(f"Key: {key} must be an integer")

        nodes = self.retrieve_nodes()

        if key >= len(nodes):
            raise SyftException(f"Index {key} out of range for retrieved nodes")

        return self.routing_client.proxy_to(nodes[key])
