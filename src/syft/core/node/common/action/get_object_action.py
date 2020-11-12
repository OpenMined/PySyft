# stdlib
from collections import OrderedDict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
from nacl.signing import VerifyKey

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectAction as GetObjectAction_PB,
)
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectResponseMessage as GetObjectResponseMessage_PB,
)
from .....proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from ..service.auth import AuthorizationException
from .common import ImmediateActionWithReply


class GetObjectResponseMessage(ImmediateSyftMessageWithoutReply):
    """
    GetObjectResponseMessages are the type of messages that are sent in reponse to a
    :class:`GetObjectAction`. They contain the object that was asked for.

    Attributes:
         obj: the object being sent back to the asker.
    """

    def __init__(
        self, obj: StorableObject, address: Address, msg_id: Optional[UID] = None
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.obj = obj

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetObjectResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetObjectResponseMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        # TODO: Fix this hack
        if isinstance(self.obj, OrderedDict):
            # convert the OrderedDict to a normal dict and then a Dict
            # syft relative
            from .....lib.python.primitive_factory import PrimitiveFactory

            sy_dict = PrimitiveFactory.generate_primitive(value=dict(self.obj))
            ser = sy_dict.serialize()
        else:
            ser = self.obj.serialize()

        # TODO: Fix this hack
        # we need to check if the serialize chain creates a storable if not
        # we need to go use the serializable_wrapper_type
        # this is because klasses have a different def serialize to normal serializables
        # which checks for the serializable_wrapper_type and uses it
        if not isinstance(ser, StorableObject_PB):
            if hasattr(self.obj, "serializable_wrapper_type"):
                obj = self.obj.serializable_wrapper_type(value=self.obj)  # type: ignore
                ser = obj.serialize()
            else:
                raise Exception(f"Cannot send {type(self.obj)} as StorableObject")

        return GetObjectResponseMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            obj=ser,
        )

    @staticmethod
    def _proto2object(proto: GetObjectResponseMessage_PB) -> "GetObjectResponseMessage":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetObjectResponseMessage
        :rtype: GetObjectResponseMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetObjectResponseMessage(
            obj=_deserialize(blob=proto.obj),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
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

        return GetObjectResponseMessage_PB


class GetObjectAction(ImmediateActionWithReply):
    """
    This kind of action is used when a Node wants to get an object located on another Node.

    The Node receiving this action first check that the asker does have the permission to
    fetch the object he asked for. If it's the case, a :class:`GetObjectResponseMessage`
    containing the object is sent back to the asker.

    Attributes:
         id_at_location: the pointer id of the object asked for.
    """

    def __init__(
        self,
        id_at_location: UID,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
        delete_obj: bool = True,
    ):
        self.id_at_location = id_at_location
        self.delete_obj = delete_obj

        # the logger needs self.id_at_location to be set already - so we call this later
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:
        try:
            try:
                storeable_object = node.store[self.id_at_location]
            except Exception as e:
                log = (
                    f"Unable to Get Object with ID {self.id_at_location} from store. "
                    + f"Possible dangling Pointer. {e}"
                )

                raise Exception(log)

            # if you are not the root user check if your verify_key has read_permission
            if (
                verify_key != node.root_verify_key
                and verify_key not in storeable_object.read_permissions
            ):
                log = (
                    f"You do not have permission to .get() Object with ID: {self.id_at_location}"
                    + "Please submit a request."
                )
                raise AuthorizationException(log)

            obj = storeable_object.data
            msg = GetObjectResponseMessage(obj=obj, address=self.reply_to, msg_id=None)

            if self.delete_obj:
                try:
                    # TODO: send EventualActionWithoutReply to delete the object at the node's
                    # convenience instead of definitely having to delete it now
                    logger.debug(
                        f"Calling delete on Object with ID {self.id_at_location} in store."
                    )
                    node.store.delete(key=self.id_at_location)
                except Exception as e:
                    log = (
                        f"> GetObjectAction delete exception {self.id_at_location} {e}"
                    )
                    logger.critical(log)
            else:
                logger.debug(f"Copying Object with ID {self.id_at_location} in store.")

            logger.debug(
                f"Returning Object with ID: {self.id_at_location} {type(storeable_object.data)}"
            )
            return msg
        except Exception as e:
            logger.error(e)
            raise e

    @property
    def pprint(self) -> str:
        return f"GetObjectAction({self.id_at_location})"

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetObjectAction_PB:
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
        return GetObjectAction_PB(
            id_at_location=self.id_at_location.proto(),
            msg_id=self.id.proto(),
            address=self.address.proto(),
            reply_to=self.reply_to.proto(),
            delete_obj=self.delete_obj,
        )

    @staticmethod
    def _proto2object(proto: GetObjectAction_PB) -> "GetObjectAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetObjectAction
        :rtype: GetObjectAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetObjectAction(
            id_at_location=_deserialize(blob=proto.id_at_location),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
            delete_obj=proto.delete_obj,
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

        return GetObjectAction_PB
