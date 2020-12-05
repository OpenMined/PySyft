# stdlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft relative
from ..core.common.message import EventualSyftMessageWithoutReply
from ..core.common.message import ImmediateSyftMessageWithReply
from ..core.common.message import ImmediateSyftMessageWithoutReply
from ..core.common.message import SignedImmediateSyftMessageWithReply
from ..core.common.message import SignedImmediateSyftMessageWithoutReply
from ..core.common.message import SyftMessage
from ..core.io.address import Address
from ..core.io.connection import ClientConnection
from ..core.io.location.specific import SpecificLocation
from ..core.io.route import SoloRoute
from ..core.node.common.client import Client
from ..core.node.device.client import DeviceClient
from ..core.node.domain.client import DomainClient
from ..core.node.network.client import NetworkClient
from ..core.node.vm.client import VirtualMachineClient
from ..decorators.syft_decorator_impl import syft_decorator


def connect(
    url: str,
    conn_type: ClientConnection,
    client_type: Client,
    credentials: Dict = {},
) -> Any:
    class GridClient(client_type):  # type: ignore
        def __init__(
            self,
            credentials: Dict,
            url: str,
            conn_type: ClientConnection,
            client_type: Client,
        ) -> None:
            # Use Server metadata
            # to build client route
            conn = conn_type(url=url)  # type: ignore

            perform_login = issubclass(client_type, DomainClient) or issubclass(  # type: ignore
                client_type, NetworkClient  # type: ignore
            )
            if perform_login:
                metadata, user_key = conn.login(credentials=credentials)
            else:
                metadata, user_key = conn._get_metadata()

            user_key = SigningKey(user_key.encode("utf-8"), encoder=HexEncoder)
            (
                spec_location,
                name,
                client_id,
            ) = client_type.deserialize_client_metadata_from_node(metadata=metadata)

            # Create a new Solo Route using the selected connection type
            route = SoloRoute(destination=spec_location, connection=conn)

            location_args = self.__route_client_location(
                client_type=client_type, location=spec_location
            )

            self.proxy_address: Optional[Address] = None

            # Create a new client using the selected client type
            super().__init__(
                network=location_args[NetworkClient],
                domain=location_args[DomainClient],
                device=location_args[DeviceClient],
                vm=location_args[VirtualMachineClient],
                name=name,
                routes=[route],
                signing_key=user_key,
            )

        @syft_decorator(typechecking=True)
        def send_immediate_msg_with_reply(
            self,
            msg: Union[
                SignedImmediateSyftMessageWithReply, ImmediateSyftMessageWithReply
            ],
            route_index: int = 0,
        ) -> SyftMessage:

            if self.proxy_address:
                msg.address = self.proxy_address

            return super(GridClient, self).send_immediate_msg_with_reply(
                msg=msg, route_index=route_index
            )

        @syft_decorator(typechecking=True)
        def proxy(self, vm_address: Address) -> None:
            self.proxy_address = vm_address

        @syft_decorator(typechecking=True)
        def send_immediate_msg_without_reply(
            self,
            msg: Union[
                SignedImmediateSyftMessageWithoutReply, ImmediateSyftMessageWithoutReply
            ],
            route_index: int = 0,
        ) -> None:
            if self.proxy_address:
                msg.address = self.proxy_address

            return super(GridClient, self).send_immediate_msg_without_reply(
                msg=msg, route_index=route_index
            )

        @syft_decorator(typechecking=True)
        def send_eventual_msg_without_reply(
            self, msg: EventualSyftMessageWithoutReply, route_index: int = 0
        ) -> None:
            if self.proxy_address:
                msg.address = self.proxy_address

            return super(GridClient, self).send_eventual_msg_without_reply(
                msg=msg, route_index=route_index
            )

        @syft_decorator(typechecking=True)
        def __route_client_location(
            self, client_type: Any, location: SpecificLocation
        ) -> Dict:
            locations: Dict[Any, Optional[SpecificLocation]] = {
                NetworkClient: None,
                DomainClient: None,
                DeviceClient: None,
                VirtualMachineClient: None,
            }
            locations[client_type] = location
            return locations

    return GridClient(
        credentials=credentials, url=url, conn_type=conn_type, client_type=client_type
    )
