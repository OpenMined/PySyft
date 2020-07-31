from ..common.client import Client
from typing import final
from ....decorators import syft_decorator
from ....common.uid import UID
from ...io.address import Address
from ...io.connection import ClientConnection
from ...io.route import Route
from typing import List


@final
class NetworkClient(Client):
    @syft_decorator(typechecking=True)
    def __init__(self, address: Address, name: str, routes: List[Route]):
        super().__init__(address=address, name=name, routes=routes)

    def add_me_to_my_address(self):
        # I should already be added
        assert self.network_id is not None

    @property
    def domain_id(self) -> UID:
        """This client points to a node, if that node lives within a domain
        or is a domain itself, this property will return the ID of that domain
        if it is known by the client."""

        raise Exception("This client points to a network, you don't have a Domain ID.")

    @domain_id.setter
    def domain_id(self, new_domain_id: UID) -> UID:
        """This client points to a node, if that node lives within a domain
        or is a domain itself and we learn the id of that domain, this setter
        allows us to save the id of that domain for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        raise Exception("This client points to a network, you don't need a Domain ID.")

    @property
    def device_id(self) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the ID of that device
        if it is known by the client."""
        raise Exception("This client points to a network, you don't have a Device ID.")

    @device_id.setter
    def device_id(self, new_device_id: UID) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the id of that device, this setter
        allows us to save the id of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        raise Exception("This client points to a network, you don't need a Device ID.")

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

        raise Exception("This client points to a network, you don't need a VM ID.")

    # @property
    # def target_node_id(self) -> UID:
    #     """This client points to a vm, this returns the id of that vm."""
    #     return self.network_id
    #
    # @target_node_id.setter
    # def target_node_id(self, new_target_node_id: UID) -> UID:
    #     """This client points to a vm, this saves the id of that vm"""
    #     self.network_id = new_target_node_id
    #     return self.network_id

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        out = f"<Network id:{self.name}>"
        return out
