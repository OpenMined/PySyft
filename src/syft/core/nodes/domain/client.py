from ..common.client import Client
from syft.core.common.uid import UID
from ...io.route import Route
from typing import List


class DomainClient(Client):
    def __init__(self, address, name, routes: List[Route]):
        super().__init__(address=address, name=name, routes=routes)

    def add_me_to_my_address(self):
        # I should already be added
        assert self.domain_id is not None

    @property
    def device_id(self) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the ID of that device
        if it is known by the client."""
        raise Exception("This client points to a domain, you don't have a Device ID.")

    @device_id.setter
    def device_id(self, new_device_id: UID) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the id of that device, this setter
        allows us to save the id of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        raise Exception("This client points to a domain, you don't need a Device ID.")

    @property
    def vm_id(self) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself, this property will return the ID of that vm
        if it is known by the client."""

        raise Exception("This client points to a network, you don't have a VM ID.")

    @vm_id.setter
    def vm_id(self, new_vm_id: UID) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        raise Exception("This client points to a device, you don't need a VM ID.")

    def __repr__(self):
        return f"<DomainClient id:{self.domain_id}>"
