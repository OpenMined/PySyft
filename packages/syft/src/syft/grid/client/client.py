# stdlib
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
import names
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
from ..messages.setup_messages import GetSetUpMessage
from ..messages.transfer_messages import LoadObjectMessage
from .enums import PyGridClientEnums
from .enums import RequestAPIFields
from .exceptions import RequestAPIException
from .grid_connection import GridHTTPConnection
from .request_api.association_api import AssociationRequestAPI
from .request_api.dataset_api import DatasetRequestAPI
from .request_api.group_api import GroupRequestAPI
from .request_api.role_api import RoleRequestAPI
from .request_api.user_api import UserRequestAPI
from .request_api.worker_api import WorkerRequestAPI

DEFAULT_PYGRID_PORT = 5000
DEFAULT_PYGRID_ADDRESS = f"http://127.0.0.1:{DEFAULT_PYGRID_PORT}"
INSECURE_PYGRID_TOKEN = "9G9MJ06OQH"  # nosec


class GridClient(DomainClient):
    def __init__(
        self,
        credentials: Dict,
        url: str,
        conn_type: Type[ClientConnection],
        client_type: Type[Client],
        user_key: Optional[SigningKey] = None,
    ) -> None:

        # Use Server metadata
        # to build client route
        self.conn = conn_type(url=url)  # type: ignore
        self.client_type = client_type

        if credentials:
            metadata, _user_key = self.conn.login(credentials=credentials)  # type: ignore
            _user_key = SigningKey(_user_key.encode(), encoder=HexEncoder)
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
        ) = self.client_type.deserialize_client_metadata_from_node(metadata=metadata)

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
        self.datasets = DatasetRequestAPI(
            send=self.__perform_grid_request, conn=self.conn, client=self
        )

    def load(
        self, obj_ptr: Type[Pointer], address: Address, pointable: bool = False
    ) -> None:
        content = {
            RequestAPIFields.ADDRESS: serialize(address)
            .SerializeToString()  # type: ignore
            .decode(PyGridClientEnums.ENCODING),
            RequestAPIFields.UID: str(obj_ptr.id_at_location.value),
            RequestAPIFields.POINTABLE: pointable,
        }
        self.__perform_grid_request(grid_msg=LoadObjectMessage, content=content)

    def setup(self, *, domain_name: Optional[str], **kwargs: Any) -> Any:

        if domain_name is None:
            domain_name = names.get_full_name() + "'s Domain"
            logging.info(
                "No Domain Name provided... picking randomly as: " + domain_name
            )

        if "token" not in kwargs.keys():
            kwargs["token"] = INSECURE_PYGRID_TOKEN
            logging.info(
                "No token provided... trying PyGrid Default (THIS IS NOT SAFE!!!): "
                + kwargs["token"]
            )

        kwargs["domain_name"] = domain_name

        response = self.conn.setup(**kwargs)  # type: ignore
        logging.info(response[RequestAPIFields.MESSAGE])

    def get_setup(self, **kwargs: Any) -> Any:
        return self.__perform_grid_request(grid_msg=GetSetUpMessage, content=kwargs)

    def search(self, query: List, pandas: bool = False) -> Any:
        response = self.__perform_grid_request(
            grid_msg=NetworkSearchMessage, content={RequestAPIFields.QUERY: query}
        )
        if pandas:
            response = DataFrame(response)

        return response

    def send_immediate_msg_with_reply(
        self,
        msg: Union[SignedImmediateSyftMessageWithReply, ImmediateSyftMessageWithReply],
        route_index: int = 0,
    ) -> Union[
        SignedImmediateSyftMessageWithoutReply, ImmediateSyftMessageWithoutReply
    ]:
        return super(GridClient, self).send_immediate_msg_with_reply(  # type: ignore
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
        # print("got this intermediary", response, type(response))
        return self.__process_response(response=response)

    def __build_msg(self, grid_msg: Any, content: Dict[Any, Any]) -> Any:
        args = {
            RequestAPIFields.ADDRESS: self.address,
            RequestAPIFields.CONTENT: content,
            RequestAPIFields.REPLY_TO: self.address,
        }
        return grid_msg(**args).sign(signing_key=self.signing_key)

    def __process_response(self, response: SyftMessage) -> Dict[Any, Any]:
        # print("calling __process_response", response, type(response))
        if response.status_code == 200:  # type: ignore
            return response.content  # type: ignore
        else:
            raise RequestAPIException(message=response.content[RequestAPIFields.ERROR])  # type: ignore


def connect(
    url: str = DEFAULT_PYGRID_ADDRESS,
    conn_type: Type[ClientConnection] = GridHTTPConnection,
    credentials: Dict = {},
    user_key: Optional[SigningKey] = None,
) -> GridClient:
    return GridClient(
        credentials=credentials,
        url=url,
        conn_type=conn_type,
        client_type=DomainClient,
        user_key=user_key,
    )


def login(
    email: Optional[str] = None,
    password: Optional[str] = None,
    url: str = DEFAULT_PYGRID_ADDRESS,
    conn_type: Type[ClientConnection] = GridHTTPConnection,
) -> GridClient:

    if email is None or password is None:
        credentials = {}
        logging.info(
            "\n\nNo email and password defined in login() - connecting as anonymous user!!!\n"
        )
    else:
        credentials = {"email": email, "password": password}

    return connect(url=url, credentials=credentials, conn_type=conn_type)
