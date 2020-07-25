from ...message.syft_message import SyftMessageWithReply
from ...message.syft_message import SyftMessageWithoutReply
from ....decorators import syft_decorator
from ....common.id import UID
from ...io.abstract import ClientConnection
from ...io.address import Address
from ...io.address import address
from ..common.node import AbstractNodeClient


class Client(AbstractNodeClient):
    @syft_decorator(typechecking=True)
    def __init__(self, target_node_id: UID, name: str, connection: ClientConnection):

        # this client points to a node, if that node lives within a network,
        # or is a network itself, this property will store the ID of that network
        # if it is known.
        self._network_id = None

        # this client points to a node, if that node lives within a domain
        # or is a domain itself, this property will store the ID of that domain
        # if it is known.
        self._domain_id = None

        # this client points to a node, if that node lives within a device
        # or is a device itself, this property will store the ID of that device
        # if it is known
        self._device_id = None

        # this client points to a node, if that node lives within a vm
        # or is a vm itself, this property will store the ID of that vm if it
        # is known
        self._vm_id = None

        self._node_address = address(
            network=self._network_id,
            domain=self._domain_id,
            device=self._device_id,
            vm=self._vm_id,
        )

        self.target_node_id = target_node_id
        self.name = name
        self.connection = connection

    @property
    def target_node_id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError

    @target_node_id.setter
    def target_node_id(self, new_target_node_id: UID) -> UID:
        """This client points to an node, this saves the id of that node"""
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
        self._node_address.pub_address.network = new_network_id
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
        self._node_address.pub_address.domain = new_domain_id
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
        self._node_address.pri_address.device = new_device_id
        return self._device_id

    @property
    def vm_id(self) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself, this property will return the ID of that vm
        if it is known by the client."""

        return self._device_id

    @vm_id.setter
    def vm_id(self, new_vm_id: UID) -> UID:
        """This client points to an node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._vm_id = new_vm_id
        self._node_address.pri_address.vm = new_vm_id
        return self._vm_id

    @property
    def node_address(self) -> Address:
        """Returns the address to use when sending messages from this client to the node.
        If we later learn more address information we can add it, but it's not required
        to be complete in all cases."""
        return self._node_address

    @syft_decorator(typechecking=True)
    def send_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        return self.connection.send_msg_with_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        return self.connection.send_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        return f"<Client pointing to node with id:{self.node_id}>"
