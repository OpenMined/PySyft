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
from ..messages.group_messages import CreateGroupMessage
from ..messages.group_messages import DeleteGroupMessage
from ..messages.group_messages import GetGroupMessage
from ..messages.group_messages import GetGroupsMessage
from ..messages.group_messages import UpdateGroupMessage
from ..messages.infra_messages import CreateWorkerMessage
from ..messages.infra_messages import DeleteWorkerMessage
from ..messages.infra_messages import GetWorkerMessage
from ..messages.infra_messages import GetWorkersMessage
from ..messages.infra_messages import UpdateWorkerMessage
from ..messages.role_messages import CreateRoleMessage
from ..messages.role_messages import DeleteRoleMessage
from ..messages.role_messages import GetRoleMessage
from ..messages.role_messages import GetRolesMessage
from ..messages.role_messages import UpdateRoleMessage
from ..messages.setup_messages import CreateInitialSetUpMessage
from ..messages.setup_messages import GetSetUpMessage
from ..messages.user_messages import CreateUserMessage
from ..messages.user_messages import DeleteUserMessage
from ..messages.user_messages import GetUserMessage
from ..messages.user_messages import GetUsersMessage
from ..messages.user_messages import UpdateUserMessage


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

        @property
        def users(self) -> Any:
            return self.__perform_grid_request(grid_msg=GetUsersMessage)

        @property
        def groups(self) -> Any:
            return self.__perform_grid_request(grid_msg=GetGroupsMessage)

        @property
        def workers(self) -> Any:
            return self.__perform_grid_request(grid_msg=GetWorkersMessage)

        @property
        def roles(self) -> Any:
            return self.__perform_grid_request(grid_msg=GetRolesMessage)

        def proxy(self, vm_address: Address) -> None:
            self.proxy_address = vm_address

        def unproxy(self) -> None:
            self.proxy_address = None

        # User API
        def create_user(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=CreateUserMessage, content=kwargs
            )

        def get_user(self, **kwargs) -> Any:
            return self.__perform_grid_request(grid_msg=GetUserMessage, content=kwargs)

        def set_user(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=UpdateUserMessage, content=kwargs
            )

        def delete_user(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=DeleteUserMessage, content=kwargs
            )

        # Group API
        def create_group(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=CreateGroupMessage, content=kwargs
            )

        def get_groups(self, **kwargs) -> Any:
            return self.__perform_grid_request(grid_msg=GetGroupMessage, content=kwargs)

        def set_group(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=UpdateGroupMessage, content=kwargs
            )

        def delete_group(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=DeleteGroupMessage, content=kwargs
            )

        # Worker API
        def create_worker(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=CreateWorkerMessage, content=kwargs
            )

        def get_worker(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=GetWorkerMessage, content=kwargs
            )

        def set_worker(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=UpdateWorkerMessage, content=kwargs
            )

        def delete_worker(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=DeleteWorkerMessage, content=kwargs
            )

        # Role API
        def create_role(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=CreateRoleMessage, content=kwargs
            )

        def get_role(self, **kwargs) -> Any:
            return self.__perform_grid_request(grid_msg=GetRoleMessage, content=kwargs)

        def set_role(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=UpdateRoleMessage, content=kwargs
            )

        def delete_role(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=DeleteRoleMessage, content=kwargs
            )

        def initial_setup(self, **kwargs) -> Any:
            return self.__perform_grid_request(
                grid_msg=CreateInitialSetUpMessage, content=kwargs
            )

        def get_setup(self, **kwargs) -> Any:
            return self.__perform_grid_request(grid_msg=GetSetUpMessage, content=kwargs)

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
            self, msg: EventualSyftMessageWithoutReply, route_index: int = 0
        ) -> None:
            if self.proxy_address:
                msg.address = self.proxy_address

            return super(GridClient, self).send_eventual_msg_without_reply(
                msg=msg, route_index=route_index
            )

        def __perform_grid_request(
            self, grid_msg: Any, content: Dict[Any, Any] = None
        ) -> Dict[Any, Any]:
            signed_msg = self.__build_msg(grid_msg=grid_msg, content=content)
            response = self.send_immediate_msg_with_reply(msg=signed_msg)
            return self.__process_response(response=response)

        def __build_msg(self, grid_msg: Any, content=Dict[Any, Any]) -> Any:
            args = {
                "address": self.address,
                "content": content,
                "reply_to": self.address,
            }
            return grid_msg(**args).sign(signing_key=self.signing_key)

        def __process_response(self, response=SyftMessage) -> Dict[Any, Any]:
            if response.status_code == 200:
                return response.content
            else:
                return Exception(response.content["error"])

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

    return GridClient(
        credentials=credentials, url=url, conn_type=conn_type, client_type=client_type
    )
