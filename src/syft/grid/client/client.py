# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from pandas import DataFrame

# syft relative
from ...core.common.message import EventualSyftMessageWithoutReply
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.common.serde.serialize import _serialize as serialize  # noqa: F401
from ...core.io.address import Address
from ...core.io.connection import ClientConnection
from ...core.io.location.specific import SpecificLocation
from ...core.io.route import SoloRoute
from ...core.node.common.client import Client
from ...core.node.device.client import DeviceClient
from ...core.node.domain.client import DomainClient
from ...core.node.network.client import NetworkClient
from ...core.node.vm.client import VirtualMachineClient
from ...core.pointer.pointer import Pointer
from ..messages.network_search_message import NetworkSearchMessage
from ..messages.setup_messages import CreateInitialSetUpMessage
from ..messages.setup_messages import GetSetUpMessage
from ..messages.transfer_messages import LoadObjectMessage
from .request_api.association_api import AssociationRequestAPI
from .request_api.group_api import GroupRequestAPI
from .request_api.role_api import RoleRequestAPI
from .request_api.user_api import UserRequestAPI
from .request_api.worker_api import WorkerRequestAPI


def connect(
    url: str,
    conn_type: Type[ClientConnection],
    credentials: Dict = {},
    user_key: Optional[SigningKey] = None,
) -> Any:
    class GridClient(DomainClient):
        def __init__(
            self,
            credentials: Dict,
            url: str,
            conn_type: Type[ClientConnection],
            client_type: Type[Client],
        ) -> None:

            # Use Server metadata
            # to build client route
            self.conn = conn_type(url=url)  # type: ignore
            self.client_type = client_type

            if credentials:
                metadata, _user_key = self.conn.login(credentials=credentials)  # type: ignore
                _user_key = SigningKey(_user_key.encode("utf-8"), encoder=HexEncoder)
            else:
                metadata = self.conn._get_metadata()  # type: ignore
                if not user_key:
                    _user_key = SigningKey.generate()
                else:
                    _user_key = user_key

            (
                spec_location,
                name,
                client_id,
            ) = self.client_type.deserialize_client_metadata_from_node(
                metadata=metadata
            )

            # Create a new Solo Route using the selected connection type
            route = SoloRoute(destination=spec_location, connection=self.conn)

            location_args = self.__route_client_location(
                client_type=self.client_type, location=spec_location
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
                signing_key=_user_key,
            )

            self.groups = GroupRequestAPI(send=self.__perform_grid_request)
            self.users = UserRequestAPI(send=self.__perform_grid_request)
            self.roles = RoleRequestAPI(send=self.__perform_grid_request)
            self.workers = WorkerRequestAPI(
                send=self.__perform_grid_request, domain_client=self
            )
            self.association_requests = AssociationRequestAPI(
                send=self.__perform_grid_request
            )

        def load(
            self, obj_ptr: Type[Pointer], address: Address, searchable: bool = False
        ) -> None:
            content = {
                "address": serialize(address).SerializeToString().decode("ISO-8859-1"),  # type: ignore
                "uid": str(obj_ptr.id_at_location.value),
                "searchable": searchable,
            }
            self.__perform_grid_request(grid_msg=LoadObjectMessage, content=content)

        def initial_setup(self, **kwargs: Any) -> Any:
            return self.__perform_grid_request(
                grid_msg=CreateInitialSetUpMessage, content=kwargs
            )

        def get_setup(self, **kwargs: Any) -> Any:
            return self.__perform_grid_request(grid_msg=GetSetUpMessage, content=kwargs)

        def search(self, query: List, pandas: bool = False) -> Any:
            response = self.__perform_grid_request(
                grid_msg=NetworkSearchMessage, content={"query": query}
            )
            if pandas:
                response = DataFrame(response)

            return response

        def send_immediate_msg_with_reply(
            self,
            msg: Union[
                SignedImmediateSyftMessageWithReply, ImmediateSyftMessageWithReply
            ],
            route_index: int = 0,
        ) -> SyftMessage:
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
            super(GridClient, self).send_immediate_msg_without_reply(
                msg=msg, route_index=route_index
            )

        def send_eventual_msg_without_reply(
            self,
            msg: EventualSyftMessageWithoutReply,
            route_index: int = 0,
        ) -> None:
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
            self, grid_msg: Any, content: Optional[Dict[Any, Any]] = None
        ) -> Dict[Any, Any]:
            if content is None:
                content = {}
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
        credentials=credentials, url=url, conn_type=conn_type, client_type=DomainClient
    )
