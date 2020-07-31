from ....common.uid import UID
from ...io.address import Address
from ...io.address import address as create_address


class LocationAwareObject:
    def __init__(self, address: Address = None):

        # All nodes should have a representation of where they think
        # they are currently held. Note that this is at risk of going
        # out of date and so we need to make sure we write good
        # logic to keep these addresses up to date. The main
        # way that it could go out of date is by the node being moved
        # by its parent or its parent being moved by a grandparent, etc.
        # without anyone telling this node. This would be bad because
        # it would mean that when the node creates a new Client for
        # someone to use, it might have trouble actually reaching
        # the node. Fortunately, the creation of a client is (always?)
        # going to be initiated by the parent node itself, so we should
        # be able to check for it there. TODO: did we check for it?

        if address is None:
            address = create_address(network=None, domain=None, device=None, vm=None)

        self._address = address

        # this address points to a node, if that node lives within a network,
        # or is a network itself, this property will store the ID of that network
        # if it is known.
        self._network_id = address.pub_address.network

        # this address points to a node, if that node lives within a domain
        # or is a domain itself, this property will store the ID of that domain
        # if it is known.
        self._domain_id = address.pub_address.domain

        # this address points to a node, if that node lives within a device
        # or is a device itself, this property will store the ID of that device
        # if it is known
        self._device_id = address.pri_address.device

        # this client points to a node, if that node lives within a vm
        # or is a vm itself, this property will store the ID of that vm if it
        # is known
        self._vm_id = address.pri_address.vm

        # make sure address includes my own ID
        self.add_me_to_my_address()

    def add_me_to_my_address(self):
        raise NotImplementedError

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

    @address.setter
    def address(self, new_address: Address) -> Address:
        raise Exception(
            "We did not design the address attribute to be updated in this"
            " way. Please update individual attributes (.vm_id, .device_id, "
            " .domain_id, or .network_id) directly. If you absolutely must,"
            " you can update ._address but we don't recommend this."
        )
