from ...message.syft_message import SyftMessageWithReply
from ...message.syft_message import SyftMessageWithoutReply
from ....decorators import syft_decorator
from ....common.id import UID
from ...io.abstract import ClientConnection
from ...io.address import Address
from ...io.address import address as create_address
from ..common.node import AbstractNodeClient
from .location_aware_object import LocationAwareObject

class Client(AbstractNodeClient, LocationAwareObject):
    """Client is an incredibly powerful abstraction in Syft. We assume that,
    no matter where a client is, it can figure out how to communicate with
    the Node it is supposed to point to. If I send you a client I have
    with all of the metadata in it, you should have all the information
    you need to know to interact with a node (although you might not
    have permissions - clients should not store private keys)."""


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

        address = create_address(
            network=self._network_id,
            domain=self._domain_id,
            device=self._device_id,
            vm=self._vm_id,
        )

        super(LocationAwareObject).__init__(address=address)

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

    @syft_decorator(typechecking=True)
    def send_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        return self.connection.send_msg_with_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        return self.connection.send_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        return f"<Client pointing to worker with id:{self.worker_id}>"
