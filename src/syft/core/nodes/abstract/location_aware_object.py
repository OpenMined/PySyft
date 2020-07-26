from ....common.id import UID
from ...io.address import Address

class LocationAwareObject:

    def __init__(self, address:Address):
        self._address = address

    @property
    def network_id(self) -> UID:
        """This client points to a node, if that node lives within a network
        or is a network itself, this property will return the ID of that network
        if it is known by the client."""

        return self._network_id


    @network_id.setter
    def network_id(self, new_network_id: UID) -> UID:
        """This client points to a node, if that node lives within a network
        or is a network itself and we learn the id of that network, this setter
        allows us to save the id of that network for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages. That
        address object will include this information if it is available"""
        self._network_id = new_network_id
        self._address.pub_address.network = new_network_id
        return self._network_id


    @property
    def domain_id(self) -> UID:
        """This client points to a node, if that node lives within a domain
        or is a domain itself, this property will return the ID of that domain
        if it is known by the client."""

        return self._domain_id


    @domain_id.setter
    def domain_id(self, new_domain_id: UID) -> UID:
        """This client points to a node, if that node lives within a domain
        or is a domain itself and we learn the id of that domain, this setter
        allows us to save the id of that domain for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._domain_id = new_domain_id
        self._address.pub_address.domain = new_domain_id
        return self._domain_id


    @property
    def device_id(self) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the ID of that device
        if it is known by the client."""
        return self._device_id


    @device_id.setter
    def device_id(self, new_device_id: UID) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the id of that device, this setter
        allows us to save the id of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._device_id = new_device_id
        self._address.pri_address.device = new_device_id
        return self._device_id


    @property
    def vm_id(self) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself, this property will return the ID of that vm
        if it is known by the client."""

        return self._vm_id


    @vm_id.setter
    def vm_id(self, new_vm_id: UID) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._vm_id = new_vm_id
        self._address.pri_address.vm = new_vm_id
        return self._vm_id


    @property
    def address(self) -> Address:
        """Returns the address to use when sending messages from this client to the node.
        If we later learn more address information we can add it, but it's not required
        to be complete in all cases."""
        return self._address