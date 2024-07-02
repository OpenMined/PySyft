# future
from __future__ import annotations

# stdlib
from typing import TYPE_CHECKING

# relative
from ..abstract_node import NodeSideType
from ..serde.serializable import serializable
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.network.routes import NodeRouteType
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_3
from ..types.syft_object import SyftObject
from ..util.assets import load_png_base64
from ..util.notebook_ui.styles import FONT_CSS
from .api import APIModule
from .client import SyftClient
from .client import login
from .client import login_as_guest
from .protocol import SyftProtocol

if TYPE_CHECKING:
    # relative
    from ..orchestra import NodeHandle


@serializable()
class EnclaveMetadata(SyftObject):
    __canonical_name__ = "EnclaveMetadata"
    __version__ = SYFT_OBJECT_VERSION_3

    route: NodeRouteType


@serializable()
class EnclaveClient(SyftClient):
    # TODO: add widget repr for enclave client

    __api_patched = False

    @property
    def code(self) -> APIModule | None:
        if self.api.has_service("code"):
            res = self.api.services.code
            # the order is important here
            # its also important that patching only happens once
            if not self.__api_patched:
                self._request_code_execution = res.request_code_execution
                self.__api_patched = True
            res.request_code_execution = self.request_code_execution
            return res
        return None

    @property
    def requests(self) -> APIModule | None:
        if self.api.has_service("request"):
            return self.api.services.request
        return None

    def connect_to_gateway(
        self,
        via_client: SyftClient | None = None,
        url: str | None = None,
        port: int | None = None,
        handle: NodeHandle | None = None,
        email: str | None = None,
        password: str | None = None,
        protocol: str | SyftProtocol = SyftProtocol.HTTP,
    ) -> SyftSuccess | SyftError | None:
        if isinstance(protocol, str):
            protocol = SyftProtocol(protocol)

        if via_client is not None:
            client = via_client
        elif handle is not None:
            client = handle.client
        else:
            client = (
                login_as_guest(url=url, port=port)
                if email is None
                else login(url=url, port=port, email=email, password=password)
            )
            if isinstance(client, SyftError):
                return client

        self.metadata: NodeMetadataJSON = self.metadata
        res = self.exchange_route(client, protocol=protocol)

        if isinstance(res, SyftSuccess):
            if self.metadata:
                return SyftSuccess(
                    message=(
                        f"Connected {self.metadata.node_type} "
                        f"'{self.metadata.name}' to gateway '{client.name}'. "
                        f"{res.message}"
                    )
                )
            else:
                return SyftSuccess(message=f"Connected to '{client.name}' gateway")

        return res

    def get_enclave_metadata(self) -> EnclaveMetadata:
        return EnclaveMetadata(route=self.connection.route)

    def _repr_html_(self) -> str:
        commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;
        .request_code_execution</span> - submit code to enclave for execution</li>
        """

        command_list = f"""
        <ul style='padding-left: 1em;'>
            {commands}
        </ul>
        """

        small_grid_symbol_logo = load_png_base64("small-grid-symbol-logo.png")

        url = getattr(self.connection, "url", None)
        node_details = f"<strong>URL:</strong> {url}<br />" if url else ""
        if self.metadata:
            node_details += f"<strong>Node Type:</strong> {self.metadata.node_type.capitalize()}<br />"
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
