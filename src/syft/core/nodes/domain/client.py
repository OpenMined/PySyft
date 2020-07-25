from ..abstract.client import Client
from ....common.id import UID
from ....decorators.syft_decorator import syft_decorator

class DomainClient(Client):
    def __init__(self, domain_id, name, connection):
        super().__init__(target_node_id=domain_id, name=name, connection=connection)

    @property
    def target_node_id(self) -> UID:
        """This client points to a domain, this returns the id of that domain."""
        return self.domain_id

    @target_node_id.setter
    def target_node_id(self, new_target_node_id: UID) -> UID:
        """This client points to a domain, this saves the id of that domain"""
        self.domain_id = new_target_node_id
        return self.domain_id

    @property
    def device_id(self) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the ID of that device
        if it is known by the client."""
        raise Exception("This client points to a domain, you don't have a Device ID.")

    @device_id.setter
    def device_id(self, new_device_id:UID) -> UID:
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
    def vm_id(self, new_vm_id:UID) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        raise Exception("This client points to a device, you don't need a VM ID.")

    def __repr__(self):
        return f"<DomainClient id:{self.id}>"
