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
from ..abstract.node import AbstractNodeClient


from .service import RequestAnswerMessageService, RequestService, RequestStatus
from .service.get_all_requests_service import GetAllRequestsService
from .service.accept_or_deny_request_service import AcceptOrDenyRequestService


class Domain(Node):
    domain: SpecificLocation
    root_key: Optional[VerifyKey]

    child_type = Device
    client_type = DomainClient
    child_type_client_type = DeviceClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: Optional[str],
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

        # specific location with name
        self.domain = SpecificLocation(name=self.name)
        self.root_key = root_key

        self.immediate_services_without_reply.append(RequestService)
        self.immediate_services_without_reply.append(AcceptOrDenyRequestService)

        self.immediate_services_with_reply.append(RequestAnswerMessageService)
        self.immediate_services_with_reply.append(GetAllRequestsService)

        self.requests = list()
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

    @syft_decorator(typechecking=True)
    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:

        # this needs to be defensive by checking domain_id NOT domain.id or it breaks
        try:
            return msg.address.domain_id == self.id and msg.address.device is None
        except Exception as e:
            error = f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}"
            print(error)
            return False

    @syft_decorator(typechecking=True)
    def set_request_status(
        self, message_request_id: UID, status: RequestStatus, client: AbstractNodeClient
    ) -> bool:
        for req in self.requests:
            if req.request_id == message_request_id:
                req.owner_client_if_available = client
                if status == RequestStatus.Accepted:
                    req.accept()
                    return True
                elif status == RequestStatus.Rejected:
                    req.deny()
                    return True

        return False

    @syft_decorator(typechecking=True)
    def get_request_status(self, message_request_id: UID) -> RequestStatus:
        # is it still pending
        for req in self.requests:
            if req.request_id == message_request_id:
                return RequestStatus.Pending

        # check if it was accepted
        # TODO remove brute search of all store objects
        # Currently theres no way to find which object to check the permissions
        # to find the stored request_id
        for obj_id in self.store.keys():
            for _, request_id in self.store[obj_id].read_permissions.items():
                if request_id == message_request_id:
                    return RequestStatus.Accepted

            for _, request_id in self.store[obj_id].search_permissions.items():
                if request_id == message_request_id:
                    return RequestStatus.Accepted

        # must have been rejected
        return RequestStatus.Rejected
