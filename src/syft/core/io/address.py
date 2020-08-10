from typing import Optional

from syft.core.common.uid import UID

from syft.decorators import syft_decorator
from syft.core.io.location import Location


# utility addresses
class All(object):
    def __repr__(self):
        return "All"


class Unspecified(object):
    def __repr__(self):
        return "Unspecified"


class Address(object):
    @syft_decorator(typechecking=True)
    def __init__(
        self,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
    ):

        # All node should have a representation of where they think
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

        # this address points to a node, if that node lives within a network,
        # or is a network itself, this property will store the ID of that network
        # if it is known.
        self._network = network

        # this address points to a node, if that node lives within a domain
        # or is a domain itself, this property will store the ID of that domain
        # if it is known.
        self._domain = domain

        # this address points to a node, if that node lives within a device
        # or is a device itself, this property will store the ID of that device
        # if it is known
        self._device = device

        # this client points to a node, if that node lives within a vm
        # or is a vm itself, this property will store the ID of that vm if it
        # is known
        self._vm = vm

    @property
    def network(self) -> UID:
        """This client points to a node, if that node lives within a network
        or is a network itself, this property will return the ID of that network
        if it is known by the client."""

        return self._network

    @network.setter
    def network(self, new_network_id: UID) -> UID:
        """This client points to a node, if that node lives within a network
        or is a network itself and we learn the id of that network, this setter
        allows us to save the id of that network for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages. That
        address object will include this information if it is available"""
        self._network = new_network_id
        return self._network

    @property
    def domain(self) -> UID:
        """This client points to a node, if that node lives within a domain
        or is a domain itself, this property will return the ID of that domain
        if it is known by the client."""

        return self._domain

    @domain.setter
    def domain(self, new_domain_id: UID) -> UID:
        """This client points to a node, if that node lives within a domain
        or is a domain itself and we learn the id of that domain, this setter
        allows us to save the id of that domain for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._domain = new_domain_id
        return self._domain

    @property
    def device(self) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the ID of that device
        if it is known by the client."""
        return self._device

    @device.setter
    def device(self, new_device_id: UID) -> UID:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the id of that device, this setter
        allows us to save the id of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._device = new_device_id
        return self._device

    @property
    def vm(self) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself, this property will return the ID of that vm
        if it is known by the client."""

        return self._vm

    @vm.setter
    def vm(self, new_vm_id: UID) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._vm = new_vm_id
        return self._vm

    @property
    def target_id(self) -> UID:
        """Return the address of the node which lives at this address.

        Note that this id is simply the most granular id available to the
        address."""
        if self._vm is not None:
            return self._vm
        elif self._device is not None:
            return self._device
        elif self._domain is not None:
            return self._domain
        elif self._network is not None:
            return self._network

        raise Exception("Address has no valid parts")

    def __repr__(self) -> str:
        out = f"<{type(self).__name__}"
        out += f" Network:{self.network},"  # OpenGrid
        out += f" Domain:{self.domain} "  # UCSF
        out += f" Device:{self.device},"  # One of UCSF's Dell Servers
        out += f" VM:{self.vm}>"  # 8GB RAM set aside @Trask - UCSF-Server-5
        return out
