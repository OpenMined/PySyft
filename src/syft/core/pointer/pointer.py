# external imports
from typing import List
from typing import Optional
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft imports
import syft as sy
from ..common.uid import UID
from ..common.pointer import AbstractPointer
from ..node.abstract.node import AbstractNode
from ..common.serde.deserialize import _deserialize
from ...decorators.syft_decorator_impl import syft_decorator
from ..node.common.action.get_object_action import GetObjectAction
from ...proto.core.pointer.pointer_pb2 import Pointer as Pointer_PB
from ..store.storeable_object import StorableObject

# TODO: Fix circular import for Client interface
# from ...core.node.common.client import Client
from typing import Any


# TODO: Fix the Client, Address, Location confusion
class Pointer(AbstractPointer):

    # automatically generated subclasses of Pointer need to be able to look up
    # the path and name of the object type they point to as a part of serde
    path_and_name: str

    def __init__(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        tags: List[str] = [],
        description: str = "",
    ) -> None:
        if id_at_location is None:
            id_at_location = UID()

        self.client = client
        self.id_at_location = id_at_location
        self.tags = tags
        self.description = description

    def get(self) -> StorableObject:
        obj_msg = GetObjectAction(
            obj_id=self.id_at_location,
            address=self.client.address,
            reply_to=self.client.address,
        )
        response = self.client.send_immediate_msg_with_reply(msg=obj_msg)

        return response.obj

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Pointer_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: Pointer_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return Pointer_PB(
            points_to_object_with_path=self.path_and_name,
            pointer_name=type(self).__name__,
            id_at_location=self.id_at_location.serialize(),
            location=self.client.address.serialize(),
            tags=self.tags,
            description=self.description,
        )

    @staticmethod
    def _proto2object(proto: Pointer_PB) -> "Pointer":
        """Creates a Pointer from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Pointer
        :rtype: Pointer

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        # TODO: we need _proto2object to include a reference to the node doing the
        # deserialization so that we can convert location into a client object. At present
        # it is an address object which will cause things to break later.

        points_to_type = sy.lib_ast(
            proto.points_to_object_with_path, return_callable=True
        )
        pointer_type = getattr(points_to_type, proto.pointer_name)
        # WARNING: This is sending a serialized Address back to the constructor
        # which currently depends on a Client for send_immediate_msg_with_reply
        return pointer_type(
            id_at_location=_deserialize(blob=proto.id_at_location),
            client=_deserialize(blob=proto.location),
            tags=proto.tags,
            description=proto.description,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """ Return the type of protobuf object which stores a class of this type

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

        return Pointer_PB

    def request_access(self, request_name: str = "", reason: str = "",) -> None:
        from ..node.domain.service import RequestMessage

        msg = RequestMessage(
            request_name=request_name,
            request_description=reason,
            address=self.client.address,
            owner_address=self.client.address,
            object_id=self.id_at_location,
            requester_verify_key=self.client.verify_key,
        )

        self.client.send_immediate_msg_without_reply(msg=msg)

    def check_access(self, node: AbstractNode, request_id: UID) -> any:  # type: ignore
        from ..node.domain.service import (
            RequestAnswerMessage,
            # RequestAnswerResponseService,
        )

        msg = RequestAnswerMessage(
            request_id=request_id, address=self.client.address, reply_to=node.address
        )
        response = self.client.send_immediate_msg_with_reply(msg=msg)
        #
        # # this should be handled by the service by default, should be patched after 0.3.0
        # RequestAnswerResponseService.process(node=node, msg=response, verify_key=msg.)

        return response.status
