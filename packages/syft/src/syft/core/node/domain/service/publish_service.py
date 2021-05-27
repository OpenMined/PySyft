# stdlib
from typing import Dict as TypeDict
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
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import bind_protobuf
from ....common.serde.serialize import _serialize as serialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithoutReply


@bind_protobuf
@final
class PublishScalarsAction(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        id_at_location: UID,
        address: Address,
        publish_ids_at_location: List[UID],
        sigma: float,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.id_at_location = id_at_location
        self.publish_ids_at_location = publish_ids_at_location
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

        return PublishScalarsAction_PB(
            id_at_location=serialize(self.id),
            address=serialize(self.address),
            publish_ids_at_location=[
                serialize(uid) for uid in self.publish_ids_at_location
            ],
            sigma=self.sigma,
            msg_id=serialize(self.id),
        )

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

        return PublishScalarsAction(
            id_at_location=deserialize(blob=proto.id_at_location),
            address=deserialize(blob=proto.address),
            publish_ids_at_location=[
                deserialize(blob=ids) for ids in proto.publish_ids_at_location
            ],
            sigma=proto.sigma,
            msg_id=deserialize(blob=proto.msg_id),
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


class PublishScalarsService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: PublishScalarsAction, verify_key: VerifyKey
    ) -> None:
        # get scalar objects from store
        publish_objects = []
        for publish_id in msg.publish_ids_at_location:
            try:
                publish_object = node.store[publish_id]
                publish_objects.append(publish_object)
            except Exception as e:
                log = (
                    f"Unable to Get Object with ID {publish_id} from store. "
                    + f"Possible dangling Pointer. {e}"
                )
                traceback_and_raise(Exception(log))

        print("we got a list of objects to publish", publish_objects)
        # TODO: Change to no reply and store the result in store with the incoming UID
        # TODO: add verify_key to result view permissions so that it can be automatically downloaded
        # result = publish(scalars, node.acc[verify_key], msg.sigma)
        # node.acc[verify_key].get_budget() # returns a float
        result = publish(publish_objects, node.acc, msg.sigma)

        # give the caller permission to download this
        read_permissions: TypeDict[VerifyKey, UID] = {verify_key: msg.id_at_location}
        search_permissions: TypeDict[VerifyKey, Optional[UID]] = {verify_key: None}

        storable = StorableObject(
            id=msg.id_at_location,
            data=result,
            description=f"Published Scalars: {msg.id_at_location}",
            read_permissions=read_permissions,
            search_permissions=search_permissions,
        )

        node.store[msg.id_at_location] = storable

    @staticmethod
    def message_handler_types() -> List[Type[PublishScalarsAction]]:
        return [PublishScalarsAction]
