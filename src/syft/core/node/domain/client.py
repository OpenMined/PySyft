# stdlib
from typing import List
from typing import Optional
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pandas as pd

# syft relative
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ...io.route import Route
from ..common.client import Client
from .service import RequestMessage


class RequestQueueClient:
    def __init__(self, client: Client) -> None:
        self.client = client

    @property
    def requests(self) -> List[RequestMessage]:
        # syft absolute
        from syft.core.node.domain.service.get_all_requests_service import (
            GetAllRequestsMessage,
        )

        msg = GetAllRequestsMessage(
            address=self.client.address, reply_to=self.client.address
        )
        requests: List[RequestMessage] = self.client.send_immediate_msg_with_reply(
            msg=msg
        ).requests

        for request in requests:
            request.gc_enabled = False  # type: ignore
            request.owner_client_if_available = self.client

        return requests

    def get_request_id_from_object_id(self, object_id: UID) -> Optional[UID]:
        for req in self.requests:
            if req.object_id == object_id:
                return req.request_id

        return object_id

    def __getitem__(self, key: Union[str, int]) -> RequestMessage:
        if isinstance(key, str):
            for request in self.requests:
                if key == str(request.id.value):
                    return request
            raise KeyError("No such request found for string id:" + str(key))
        if isinstance(key, int):
            return self.requests[key]
        else:
            raise KeyError("Please pass in a string or int key")

    def __repr__(self) -> str:
        return repr(self.requests)

    @property
    def pandas(self) -> pd.DataFrame:
        request_lines = [
            {
                "Request Name": request.request_name,
                "Reason": request.request_description,
                "Request ID": request.id,
                "Requested Object's ID": request.object_id,
            }
            for request in self.requests
        ]
        return pd.DataFrame(request_lines)


class DomainClient(Client):

    domain: SpecificLocation
    requests: RequestQueueClient

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

        self.requests = RequestQueueClient(client=self)
        self.post_init()

    @property
    def id(self) -> UID:
        return self.domain.id

    @property
    def device(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the Location of that device
        if it is known by the client."""

        return super().device

    @device.setter
    def device(self, new_device: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the Location of that device, this setter
        allows us to save the Location of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        raise Exception("This client points to a domain, you don't need a Device ID.")

    @property
    def vm(self) -> Optional[Location]:
        """This client points to an node, if that node lives within a vm
        or is a vm itself, this property will return the Location of that vm
        if it is known by the client."""

        return super().vm

    @vm.setter
    def vm(self, new_vm: Location) -> Optional[Location]:
        """This client points to an node, if that node lives within a vm
        or is a vm itself and we learn the Location of that vm, this setter
        allows us to save the Location of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        raise Exception("This client points to a device, you don't need a VM Location.")

    def __repr__(self) -> str:
        return f"<DomainClient:{self.id}>"
