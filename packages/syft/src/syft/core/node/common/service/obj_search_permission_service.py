# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from ..... import serialize
from .....proto.core.node.common.service.object_search_permission_update_message_pb2 import (
    ObjectSearchPermissionUpdateMessage as ObjectSearchPermissionUpdateMessage_PB,
)
from ....common.group import VERIFYALL
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from ...common.service.auth import AuthorizationException
from .auth import service_auth
from .node_service import ImmediateNodeServiceWithoutReply


@bind_protobuf
@final
class ObjectSearchPermissionUpdateMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        add_instead_of_remove: bool,
        target_verify_key: Optional[VerifyKey],
        target_object_id: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

        self.add_instead_of_remove = add_instead_of_remove
        self.target_verify_key = target_verify_key
        self.target_object_id = target_object_id

    def _object2proto(self) -> ObjectSearchPermissionUpdateMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectSearchPermissionUpdateMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return ObjectSearchPermissionUpdateMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            target_verify_key=bytes(self.target_verify_key)
            if self.target_verify_key
            else None,
            target_object_id=serialize(self.target_object_id),
            add_instead_of_remove=self.add_instead_of_remove,
        )

    @staticmethod
    def _proto2object(
        proto: ObjectSearchPermissionUpdateMessage_PB,
    ) -> "ObjectSearchPermissionUpdateMessage":
        """Creates a ObjectSearchPermissionUpdateMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ObjectSearchPermissionUpdateMessage
        :rtype: ObjectSearchPermissionUpdateMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return ObjectSearchPermissionUpdateMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            target_verify_key=VerifyKey(proto.target_verify_key)
            if proto.target_verify_key
            else None,
            target_object_id=_deserialize(blob=proto.target_object_id),
            add_instead_of_remove=proto.add_instead_of_remove,
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

        return ObjectSearchPermissionUpdateMessage_PB


class ImmediateObjectSearchPermissionUpdateService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(
        node: AbstractNode,
        msg: ObjectSearchPermissionUpdateMessage,
        verify_key: VerifyKey,
    ) -> None:
        storable_object = node.store[msg.target_object_id]
        if (
            verify_key != node.root_verify_key
            or verify_key not in storable_object.read_permissions
        ):
            log = (
                f"You do not have permission to update Object with ID: {msg.target_object_id}"
                + "Please submit a request."
            )
            raise AuthorizationException(log)
        target_verify_key = msg.target_verify_key or VERIFYALL
        if msg.add_instead_of_remove:
            storable_object.search_permissions[target_verify_key] = msg.id
        else:
            storable_object.search_permissions.pop(target_verify_key, None)

    @staticmethod
    def message_handler_types() -> List[Type[ObjectSearchPermissionUpdateMessage]]:
        return [ObjectSearchPermissionUpdateMessage]
