# stdlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft relative
from ...core.common.message import EventualSyftMessageWithoutReply
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.io.address import Address
from ...core.io.connection import ClientConnection
from ...core.io.location.specific import SpecificLocation
from ...core.io.route import SoloRoute
from ...core.node.common.client import Client
from ...core.node.device.client import DeviceClient
from ...core.node.domain.client import DomainClient
from ...core.node.network.client import NetworkClient
from ...core.node.vm.client import VirtualMachineClient
from ..messages.setup_messages import CreateInitialSetUpMessage
from ..messages.setup_messages import GetSetUpMessage
from .request_api.association_api import AssociationRequestAPI
from .request_api.group_api import GroupRequestAPI
from .request_api.role_api import RoleRequestAPI
from .request_api.user_api import UserRequestAPI
from .request_api.worker_api import WorkerRequestAPI


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

            if credentials:
                metadata, user_key = conn.login(credentials=credentials)
                user_key = SigningKey(user_key.encode("utf-8"), encoder=HexEncoder)
            else:
                metadata = conn._get_metadata()
                user_key = SigningKey.generate()

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

            self.groups = GroupRequestAPI(send=self.__perform_grid_request)
            self.users = UserRequestAPI(send=self.__perform_grid_request)
            self.roles = RoleRequestAPI(send=self.__perform_grid_request)
            self.workers = WorkerRequestAPI(send=self.__perform_grid_request)
            self.association_requests = AssociationRequestAPI(
                send=self.__perform_grid_request
            )

        def proxy(self, vm_address: Address) -> None:
            self.proxy_address = vm_address

        def unproxy(self) -> None:
            self.proxy_address = None

        def initial_setup(self, **kwargs: Any) -> Any:
            return self.__perform_grid_request(
                grid_msg=CreateInitialSetUpMessage, content=kwargs
            )

        def get_setup(self, **kwargs: Any) -> Any:
            return self.__perform_grid_request(grid_msg=GetSetUpMessage, content=kwargs)

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

        def send_eventual_msg_without_reply(
            self,
            msg: EventualSyftMessageWithoutReply,
            route_index: int = 0,
        ) -> None:
            if self.proxy_address:
                msg.address = self.proxy_address

            return super(GridClient, self).send_eventual_msg_without_reply(
                msg=msg, route_index=route_index
            )

        def __route_client_location(
            self, client_type: Any, location: SpecificLocation
        ) -> Dict[Any, Any]:
            locations: Dict[Any, Optional[SpecificLocation]] = {
                NetworkClient: None,
                DomainClient: None,
                DeviceClient: None,
                VirtualMachineClient: None,
            }
            locations[client_type] = location
            return locations

        def __perform_grid_request(
            self, grid_msg: Any, content: Dict[Any, Any] = {}
        ) -> Dict[Any, Any]:
            signed_msg = self.__build_msg(grid_msg=grid_msg, content=content)
            response = self.send_immediate_msg_with_reply(msg=signed_msg)
            return self.__process_response(response=response)

        def __build_msg(self, grid_msg: Any, content: Dict[Any, Any]) -> Any:
            args = {
                "address": self.address,
                "content": content,
                "reply_to": self.address,
            }
            return grid_msg(**args).sign(signing_key=self.signing_key)

        def __process_response(self, response: SyftMessage) -> Dict[Any, Any]:
            if response.status_code == 200:  # type: ignore
                return response.content  # type: ignore
            else:
                raise Exception(response.content["error"])  # type: ignore

    return GridClient(
        credentials=credentials, url=url, conn_type=conn_type, client_type=client_type
    )
