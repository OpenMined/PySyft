# stdlib
from typing import List as TypeList

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ......proto.core.node.domain.service.pss_pb2 import PSA  # type: ignore
from .....common.message import ImmediateSyftMessageWithoutReply  # type: ignore
from .....common.serde.deserialize import _deserialize as deserialize  # type: ignore
from .....common.serde.serializable import serializable  # type: ignore
from .....common.serde.serialize import _serialize as serialize  # type: ignore
from .....common.uid import UID  # type: ignore
from .....io.address import Address  # type: ignore


@serializable()
@final
class PublishScalarsAction(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        id_at_location: UID,
        address: Address,
        publish_ids_at_location: TypeList[UID],
        sigma: float,
    ):
        super().__init__(address=address)
        self.id_at_location = id_at_location
        self.publish_ids_at_location = publish_ids_at_location
        self.sigma = sigma

    def _object2proto(self) -> PSA:
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

        return PSA(
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            publish_ids_at_location=[
                serialize(uid) for uid in self.publish_ids_at_location
            ],
            sigma=self.sigma,
        )

    @staticmethod
    def _proto2object(proto: PSA) -> "PublishScalarsAction":
        """Creates a PublishScalarsAction from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of PublishScalarsAction
        :rtype: PublishScalarsAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return PublishScalarsAction(
            id_at_location=deserialize(blob=proto.id_at_location),
            address=deserialize(blob=proto.address),
            publish_ids_at_location=[
                deserialize(blob=ids) for ids in proto.publish_ids_at_location
            ],
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

        return PSA
