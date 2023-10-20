# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import TYPE_CHECKING

# relative
from ..abstract_node import NodeSideType
from ..client.api import APIRegistry
from ..img.base64 import base64read
from ..serde.serializable import serializable
from ..service.network.routes import NodeRouteType
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID
from ..util.fonts import fonts_css
from .api import APIModule
from .client import SyftClient
from .client import login
from .client import login_as_guest

if TYPE_CHECKING:
    # relative
    from ..service.code.user_code import SubmitUserCode


@serializable()
class EnclaveMetadata(SyftObject):
    __canonical_name__ = "EnclaveMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    route: NodeRouteType


@serializable()
class EnclaveClient(SyftClient):
    # TODO: add widget repr for enclave client

    __api_patched = False

    @property
    def code(self) -> Optional[APIModule]:
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
    def requests(self) -> Optional[APIModule]:
        if self.api.has_service("request"):
            return self.api.services.request
        return None

    def connect_to_gateway(
        self,
        via_client: Optional[SyftClient] = None,
        url: Optional[str] = None,
        port: Optional[int] = None,
        handle: Optional[NodeHandle] = None,  # noqa: F821
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
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

        res = self.exchange_route(client)
        if isinstance(res, SyftSuccess):
            return SyftSuccess(
                message=f"Connected {self.metadata.node_type} to {client.name} gateway"
            )
        return res

    def get_enclave_metadata(self) -> EnclaveMetadata:
        return EnclaveMetadata(route=self.connection.route)

    def request_code_execution(self, code: SubmitUserCode):
        # relative
        from ..service.code.user_code_service import SubmitUserCode

        if not isinstance(code, SubmitUserCode):
            raise Exception(
                f"The input code should be of type: {SubmitUserCode} got:{type(code)}"
            )

        enclave_metadata = self.get_enclave_metadata()

        code_id = UID()
        code.id = code_id
        code.enclave_metadata = enclave_metadata

        apis = []
        for k, v in code.input_policy_init_kwargs.items():
            # We would need the verify key of the data scientist to be able to index the correct client
            # Since we do not want the data scientist to pass in the clients to the enclave client
            # from a UX perspecitve.
            # we will use the recent node id to find the correct client
            # assuming that it is the correct client
            # Warning: This could lead to inconsistent results, when we have multiple clients
            # in the same node pointing to the same node.
            # One way, by which we could solve this in the long term,
            # by forcing the user to pass only assets to the sy.ExactMatch,
            # by which we could extract the verify key of the data scientist
            # as each object comes with a verify key and node_uid
            # the asset object would contain the verify key of the data scientist.
            api = APIRegistry.get_by_recent_node_uid(k.node_id)
            if api is None:
                raise ValueError(f"could not find client for input {v}")
            else:
                apis += [api]

        for api in apis:
            res = api.services.code.request_code_execution(code=code)
            if isinstance(res, SyftError):
                return res

        # we are using the real method here, see the .code property getter
        _ = self.code
        res = self._request_code_execution(code=code)

        return res

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
