from syft.core.node.domain.client import DomainClient
from typing import Union
from typing import Any
from typing import Optional
from typing import List
from typing import Type
from ...core.common.message import EventualSyftMessageWithoutReply
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.node.network.client import NetworkClient
from ...core.node.domain.client import DomainClient
from ...core.node.common.client import Client
from ...core.io.route import Route
from ...core.io.route import SoloRoute
from ...core.io.location import SpecificLocation
from ...core.io.location import Location
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from ...core.common.uid import UID
from .grid_connection import GridHTTPConnection



class ProxyClient(DomainClient):
    def __init__(
        self,
        name: Optional[str],
        routes: List[Route],
        domain: SpecificLocation,
        network: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
    ):
        super().__init__(
            name=name,
            routes=routes,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        self.proxy_address = None

    def send_immediate_msg_with_reply(
        self,
        msg: Union[
            SignedImmediateSyftMessageWithReply,
            ImmediateSyftMessageWithReply,
            Any,  # TEMPORARY until we switch everything to NodeRunnableMessage types.
        ],
        route_index: int = 0,
    ) -> SyftMessage:
        if self.proxy_address:
            msg.address = self.proxy_address
        return super().send_immediate_msg_with_reply(msg=msg, route_index=route_index)

    def send_immediate_msg_without_reply(
        self,
        msg: Union[
            SignedImmediateSyftMessageWithoutReply, ImmediateSyftMessageWithoutReply
        ],
        route_index: int = 0,
    ) -> None:
        if self.proxy_address:
            msg.address = self.proxy_address
        super().send_immediate_msg_without_reply(msg=msg, route_index=route_index)

    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply, route_index: int = 0
    ) -> None:
        if self.proxy_address:
            msg.address = self.proxy_address
        super().send_eventual_msg_without_reply(msg=msg, route_index=route_index)



def proxy_connect(
    proxy_address,
    url,
    credentials = {},
    user_key: Optional[SigningKey] = None,
) -> Client:
    # Use Server metadata
    # to build client route
    conn = GridHTTPConnection(url=url)  # type: ignore

    if credentials:
        metadata, _user_key = conn.login(credentials=credentials)  # type: ignore
        _user_key = SigningKey(_user_key.encode(), encoder=HexEncoder)
    else:
        metadata = conn._get_metadata()  # type: ignore
        if not user_key:
            _user_key = SigningKey.generate()
        else:
            _user_key = user_key

    # Check node client type based on metadata response
    # client_type: Union[Type[DomainClient], Type[NetworkClient]]
    client_type = ProxyClient
    (
        spec_location,
        name,
        client_id,
    ) = client_type.deserialize_client_metadata_from_node(metadata=metadata)

    # Create a new Solo Route using the selected connection type
    route = SoloRoute(destination=spec_location, connection=conn)

    kwargs = {"name": name, "routes": [route], "signing_key": _user_key}

    kwargs["domain"] = proxy_address.domain

    # Create a new client using the selected client type
    node = client_type(**kwargs)

    # Set Domain's proxy address
    node.proxy_address = proxy_address
    return node
