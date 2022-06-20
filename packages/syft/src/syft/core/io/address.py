# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from ...logger import debug
from ...logger import traceback_and_raise
from ...proto.core.io.address_pb2 import Address as Address_PB
from ...util import key_emoji as key_emoji_util
from ...util import random_name
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import serializable
from ..common.serde.serialize import _serialize as serialize
from ..common.uid import UID
from .location import Location


class Unspecified(object):
    def __repr__(self) -> str:
        return "Unspecified"


@serializable()
class Address:
    name: Optional[str]

    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
    ):
        self.name = name if name is not None else random_name()

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
    def icon(self) -> str:
        # 4 different aspects of location
        icon = "💠"
        sub = []
        if self.vm is not None:
            sub.append("🍰")
        if self.device is not None:
            sub.append("📱")
        if self.domain is not None:
            sub.append("🏰")
        if self.network is not None:
            sub.append("🔗")

        if len(sub) > 0:
            icon = f"{icon} ["
            for s in sub:
                icon += s
            icon += "]"
        return icon

    @property
    def pprint(self) -> str:
        output = f"{self.icon} {self.name} ({str(self.__class__.__name__)})"
        if hasattr(self, "id"):
            output += f"@{self.target_id.id.emoji()}"
        return output

    def post_init(self) -> None:
        debug(f"> Creating {self.pprint}")

    def key_emoji(self, key: Union[bytes, SigningKey, VerifyKey]) -> str:
        return key_emoji_util(key=key)

    @property
    def address(self) -> "Address":
        # QUESTION what happens if we have none of these?

        # sneak the name on there
        if hasattr(self, "name"):
            name = self.name
        else:
            name = random_name()

        address = Address(
            name=name,
            network=self.network,
            domain=self.domain,
            device=self.device,
            vm=self.vm,
        )

        return address

    def _object2proto(self) -> Address_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: Address_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return Address_PB(
            name=self.name,
            has_network=self.network is not None,
            network=serialize(self.network) if self.network is not None else None,
            has_domain=self.domain is not None,
            domain=serialize(self.domain) if self.domain is not None else None,
            has_device=self.device is not None,
            device=serialize(self.device) if self.device is not None else None,
            has_vm=self.vm is not None,
            vm=serialize(self.vm) if self.vm is not None else None,
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
            name=proto.name,
            network=_deserialize(blob=proto.network) if proto.has_network else None,
            domain=_deserialize(blob=proto.domain) if proto.has_domain else None,
            device=_deserialize(blob=proto.device) if proto.has_device else None,
            vm=_deserialize(blob=proto.vm) if proto.has_vm else None,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
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
    def network_id(self) -> Optional[UID]:
        network = self.network
        if network is not None:
            return network.id
        return None

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
    def domain_id(self) -> Optional[UID]:
        domain = self.domain
        if domain is not None:
            return domain.id
        return None

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
    def device_id(self) -> Optional[UID]:
        device = self.device
        if device is not None:
            return device.id
        return None

    @property
    def vm(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a vm
        or is a vm itself, this property will return the ID of that vm
        if it is known by the client."""

        return self._vm

    @vm.setter
    def vm(self, new_vm: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._vm = new_vm
        return self._vm

    @property
    def vm_id(self) -> Optional[UID]:
        vm = self.vm
        if vm is not None:
            return vm.id
        return None

    def target_emoji(self) -> str:
        output = ""
        if self.target_id is not None:
            output = f"@{self.target_id.id.emoji()}"
        return output

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

        traceback_and_raise(Exception("Address has no valid parts"))

    def __eq__(self, other: Any) -> bool:
        """Returns whether two Address objects refer to the same set of locations

        :param other: the other object to compare with self
        :type other: Any (note this must be Any or __eq__ fails on other types)
        :returns: whether the two objects are the same
        :rtype: bool
        """

        try:
            a = self.network == other.network
            b = self.domain == other.domain
            c = self.device == other.device
            d = self.vm == other.vm

            return a and b and c and d
        except Exception:
            return False

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
