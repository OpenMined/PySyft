# external class imports
from typing import List

# syft imports
from ...io.location import SpecificLocation
from ...io.location import Location
from ..common.client import Client
from ...io.route import Route
from ...common.uid import UID
from typing import Optional


class DomainClient(Client):
    def __init__(
        self,
        name: str,
        routes: List[Route],
        domain: SpecificLocation,
        network: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
    ):
        super().__init__(
            name=name,
            routes=routes,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
        )

    @property
    def id(self) -> Optional[UID]:
        return self.domain.id if self.domain is not None else None

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
