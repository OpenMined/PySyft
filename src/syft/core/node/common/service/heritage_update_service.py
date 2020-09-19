"""The purpose of this service is to inform lower level devices
of changes in the hierarchy above them. For example, if a Domain
registers within a new Network or if a Device registers within
a new Domain, all the other child node will need to know this
information to populate complete addresses into their clients."""

# external class imports
from nacl.signing import VerifyKey
from typing import List
from google.protobuf.reflection import GeneratedProtocolMessageType

from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID

from .....proto.core.node.common.service.heritage_update_service_pb2 import (
    HeritageUpdateMessage as HeritageUpdateMessage_PB,
)

import syft as sy
from .auth import service_auth
from ....io.address import Address
from .....decorators import syft_decorator
from ...abstract.node import AbstractNode
from .node_service import ImmediateNodeServiceWithoutReply
from ....common.serde.deserialize import _deserialize

# TODO: change all old_message names in syft to have "WithReply" or "WithoutReply"
# at the end of the name


class HeritageUpdateMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self, new_ancestry_address: Address, address: Address, msg_id: UID = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.new_ancestry_address = new_ancestry_address

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> HeritageUpdateMessage_PB:
        return HeritageUpdateMessage_PB(
            new_ancestry_address=self.new_ancestry_address.serialize(),
            address=self.address.serialize(),
            msg_id=self.id.serialize(),
        )

    @staticmethod
    def _proto2object(proto: HeritageUpdateMessage_PB) -> "HeritageUpdateMessage":
        return HeritageUpdateMessage(
            new_ancestry_address=_deserialize(blob=proto.new_ancestry_address),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return HeritageUpdateMessage_PB


class HeritageUpdateService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(root_only=True)
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: HeritageUpdateMessage, verify_key: VerifyKey
    ) -> None:
        if sy.VERBOSE:
            print(
                f"> Executing {HeritageUpdateService.pprint()} {msg.pprint} on {node.pprint}"
            )
        addr = msg.new_ancestry_address

        if addr.network is not None:
            node.network = addr.network
        if addr.domain is not None:
            node.domain = addr.domain
        if addr.device is not None:
            node.device = addr.device

        for node_client in node.known_child_nodes:
            try:
                # TODO: Client (and possibly Node) should subclass from StorableObject
                location_id = node_client.data.target_id.id
                msg.address = node_client.data
                try:
                    in_memory_client = node.in_memory_client_registry[location_id]
                    # we need to sign here with the current node not the destination side
                    in_memory_client.send_immediate_msg_without_reply(msg=msg)
                    if sy.VERBOSE:
                        print(f"> Flowing {msg.pprint} to {addr.target_emoji()}")
                    return None
                except Exception as e:
                    if sy.VERBOSE:
                        print(f"{location_id} not on nodes in_memory_client. {e}")
                    pass
            except Exception as e:
                print(e)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [HeritageUpdateMessage]
