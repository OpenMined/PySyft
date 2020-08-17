# external class imports
from typing import Optional
from typing import Union
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft imports
from ....decorators.syft_decorator_impl import syft_decorator
from ...common.message import SyftMessage, SignedMessage
from ...io.location import SpecificLocation
from ..device import Device, DeviceClient
from ...io.location import Location
from .client import DomainClient
from ..common.node import Node
from ...common.uid import UID


import pandas
from typing import Dict
from .service import (
    RequestAnswerMessageService,
    RequestAnswerResponseService,
    RequestService,
    RequestMessage,
    RequestAnswerResponse,
    RequestStatus,
)


class Requests:
    _requests: Dict[UID, dict] = {}
    _object2request: Dict[UID, UID] = {}
    _responses: Dict[UID, RequestStatus] = {}

    def register_request(self, msg: RequestMessage) -> None:
        self._requests[msg.request_id] = {
            "message": msg,
            "status": RequestStatus.Pending,
        }

    def register_response(self, msg: RequestAnswerResponse) -> None:
        self._responses[msg.request_id] = msg.status

    def get_status(self, request_id: UID) -> RequestStatus:
        return self._requests[request_id]["status"]

    def register_mapping(self, object_id: UID, request_id: UID) -> None:
        self._object2request[object_id] = request_id

    def get_request_id_from_object_id(self, object_id: UID) -> UID:
        return self._object2request[object_id]

    def set_request_status(self, request_id: UID, status: RequestStatus) -> None:
        self._requests[request_id]["status"] = status

    def display_requests(self) -> pandas.DataFrame:
        request_lines = []
        for request_id, request in self._requests.items():
            request_lines.append(
                {
                    "Request name": request["message"].request_name,
                    "Request description": request["message"].request_description,
                    "Request ID": request_id,
                    "Object Status": request["status"],
                    "Object ID": request["message"].object_id,
                }
            )
        return pandas.DataFrame(request_lines)


class Domain(Node):
    domain: SpecificLocation
    root_key: Optional[VerifyKey]

    child_type = Device
    client_type = DomainClient
    child_type_client_type = DeviceClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        network: Optional[Location] = None,
        domain: SpecificLocation = SpecificLocation(),
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        root_key: Optional[VerifyKey] = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        self.root_key = root_key

        self.immediate_services_without_reply.append(RequestService)
        self.immediate_services_without_reply.append(RequestAnswerResponseService)

        self.immediate_services_with_reply.append(RequestAnswerMessageService)
        self.requests = Requests()
        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()

        self.post_init()

    @property
    def icon(self) -> str:
        return "🏰"

    @property
    def id(self) -> UID:
        return self.domain.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:

        # this needs to be defensive by checking domain_id NOT domain.id or it breaks
        try:
            return msg.address.domain_id == self.id and msg.address.device is None
        except Exception as e:
            error = f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}"
            print(error)
            return False

    def set_request_status(self, request_id: UID, status: RequestStatus) -> None:

        self.requests.set_request_status(request_id, status)
