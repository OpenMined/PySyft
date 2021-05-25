# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from .....core.adp.publish import publish
from .....logger import traceback_and_raise
from .....proto.core.node.domain.service.publish_scalars_service_pb2 import (
    PublishScalarsAction as PublishScalarsAction_PB,
)
from .....proto.core.node.domain.service.publish_scalars_service_pb2 import (
    PublishScalarsResultMessage as PublishScalarsResultMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import bind_protobuf
from ....common.serde.serialize import _serialize as serialize
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithoutReply


@bind_protobuf
@final
class PublishScalarsAction(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        scalar_ids_at_location: List[UID],
        sigma: float,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, reply_to=reply_to, msg_id=msg_id)
        self.scalar_ids_at_location = scalar_ids_at_location
        self.sigma = sigma

    def _object2proto(self) -> PublishScalarsAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: PublishScalarsAction_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        pb = PublishScalarsAction_PB(
            msg_id=serialize(self.id, to_proto=True),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
            sigma=self.sigma,
        )

        for id_obj in self.scalar_ids_at_location:
            pb.scalar_ids_at_location.append(serialize(id_obj))

        return pb

    @staticmethod
    def _proto2object(proto: PublishScalarsAction_PB) -> "PublishScalarsAction":
        """Creates a PublishScalarsAction from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of PublishScalarsAction
        :rtype: PublishScalarsAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        scalar_ids_at_location = []
        for id_at_location in proto.scalar_ids_at_location:
            scalar_ids_at_location.append(_deserialize(blob=id_at_location))

        return PublishScalarsAction(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
            scalar_ids_at_location=scalar_ids_at_location,
            sigma=proto.sigma,
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

        return PublishScalarsAction_PB


@bind_protobuf
class PublishScalarsResultMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, address: Address, result: float):
        super().__init__(address)
        self.result = result

    def _object2proto(self) -> PublishScalarsResultMessage_PB:
        msg = PublishScalarsResultMessage_PB()
        msg.result = self.result
        msg.address.CopyFrom(serialize(obj=self.address))
        return msg

    @staticmethod
    def _proto2object(
        proto: PublishScalarsResultMessage_PB,
    ) -> "PublishScalarsResultMessage":
        request_response = PublishScalarsResultMessage(
            address=_deserialize(blob=proto.address), result=proto.result
        )
        return request_response

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return PublishScalarsResultMessage_PB


class PublishScalarsService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: PublishScalarsAction, verify_key: VerifyKey
    ) -> PublishScalarsResultMessage:
        # get scalar objects from store
        scalars = []
        for scalar_id_at_location in msg.scalar_ids_at_location:
            try:
                scalar = node.store[scalar_id_at_location]
                scalars.append(scalar)
            except Exception as e:
                log = (
                    f"Unable to Get Object with ID {scalar_id_at_location} from store. "
                    + f"Possible dangling Pointer. {e}"
                )
                traceback_and_raise(Exception(log))


        result = publish(scalars, node.acc[verify_key], msg.sigma)
        # TODO: add verify_key to result view permissions so that it can be automatically downloadedd

        # return <pointer to result>

        # node.acc[verify_key].get_budget() # returns a float
        # return PublishScalarsResultMessage(address=msg.reply_to, result=result)

    @staticmethod
    def message_handler_types() -> List[Type[PublishScalarsAction]]:
        return [PublishScalarsAction]
