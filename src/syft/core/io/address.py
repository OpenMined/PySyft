# external class imports
from typing import Optional

# syft imports (sorted by length)
from ..io.location import Location
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import Serializable
from ...decorators.syft_decorator_impl import syft_decorator
from ...proto.core.io.address_pb2 import Address as Address_PB
from google.protobuf.reflection import GeneratedProtocolMessageType


# utility addresses
class All(object):
    def __repr__(self):
        return "All"


class Unspecified(object):
    def __repr__(self):
        return "Unspecified"


class Address(Serializable):
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

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Address_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectWithID_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return Address_PB(
            has_network=self.network is not None,
            network=self.network.serialize() if self.network is not None else None,
            has_domain=self.network is not None,
            domain=self.domain.serialize() if self.domain is not None else None,
            has_device=self.device is not None,
            device=self.device.serialize() if self.device is not None else None,
            has_vm=self.vm is not None,
            vm=self.vm.serialize() if self.vm is not None else None,
        )

    @staticmethod
    def _proto2object(proto: Address_PB) -> "Address":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ObjectWithID
        :rtype: ObjectWithID

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return Address(
            network=_deserialize(blob=proto.network) if proto.has_network else None,
            domain=_deserialize(blob=proto.domain) if proto.has_domain else None,
            device=_deserialize(blob=proto.device) if proto.has_device else None,
            vm=_deserialize(blob=proto.vm) if proto.has_vm else None,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """ Return the type of protobuf object which stores a class of this type

        As a part of serializatoin and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return Address_PB

    @property
    def network(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a network
        or is a network itself, this property will return the ID of that network
        if it is known by the client."""

        return self._network

    @network.setter
    def network(self, new_network: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a network
        or is a network itself and we learn the id of that network, this setter
        allows us to save the id of that network for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages. That
        address object will include this information if it is available"""
        self._network = new_network
        return self._network

    @property
    def domain(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a domain
        or is a domain itself, this property will return the ID of that domain
        if it is known by the client."""

        return self._domain

    @domain.setter
    def domain(self, new_domain: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a domain
        or is a domain itself and we learn the id of that domain, this setter
        allows us to save the id of that domain for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._domain = new_domain
        return self._domain

    @property
    def device(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the ID of that device
        if it is known by the client."""
        return self._device

    @device.setter
    def device(self, new_device: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the id of that device, this setter
        allows us to save the id of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._device = new_device
        return self._device

    @property
    def vm(self) -> Optional[Location]:
        """This client points to an node, if that node lives within a vm
        or is a vm itself, this property will return the ID of that vm
        if it is known by the client."""

        return self._vm

    @vm.setter
    def vm(self, new_vm: Location) -> Optional[Location]:
        """This client points to an node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._vm = new_vm
        return self._vm

    @property
    def target_id(self) -> Location:
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

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: "Address") -> bool:
        """Returns whether two Address objects refer to the same set of locations

        :param other: the other object to compare with self
        :param type: Address
        :returns: whether the two objects are the same
        :rtype: bool
        """

        a = self.network == other.network
        b = self.domain == other.domain
        c = self.device == other.device
        d = self.vm == other.vm

        return a and b and c and d

    def __repr__(self) -> str:
        out = f"<{type(self).__name__} -"
        if self.network is not None:
            out += f" Network:{self.network.repr_short()},"  # OpenGrid
        if self.domain is not None:
            out += f" Domain:{self.domain.repr_short()} "  # UCSF
        if self.device is not None:
            out += f" Device:{self.device.repr_short()},"  # One of UCSF's Dell Servers
        if self.vm is not None:
            out += f" VM:{self.vm.repr_short()}"  # 8GB RAM set aside @Trask - UCSF-Server-5

        # remove extraneous comma and add a close carrot
        return out[:-1] + ">"
