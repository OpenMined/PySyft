# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# relative
from .....logger import debug
from .....logger import traceback_and_raise
from .....logger import warning
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectAction as GetObjectAction_PB,
)
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectResponseMessage as GetObjectResponseMessage_PB,
)
from .....util import get_fully_qualified_name
from .....util import validate_type
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import serializable
from ....common.serde.serialize import _serialize as serialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from ..node_service.auth import AuthorizationException
from .common import ImmediateActionWithReply


@serializable()
class GetObjectResponseMessage(ImmediateSyftMessageWithoutReply):
    """
    GetObjectResponseMessages are the type of messages that are sent in response to a
    :class:`GetObjectAction`. They contain the object that was asked for.

    Attributes:
         obj: the object being sent back to the asker.
    """

    def __init__(
        self, obj: StorableObject, address: Address, msg_id: Optional[UID] = None
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.obj = obj

    def _object2proto(self) -> GetObjectResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetObjectResponseMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        ser = serialize(self.obj)

        return GetObjectResponseMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
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
            obj=deserialize(blob=proto.obj),
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
        )

    @property
    def data(self) -> object:
        data = self.obj.data
        try:
            # TODO: Make only for DataFrames etc
            # Issue: https://github.com/OpenMined/PySyft/issues/5322
            if get_fully_qualified_name(obj=self.obj.data) not in [
                "pandas.core.frame.DataFrame",
                "pandas.core.series.Series",
            ]:
                data.tags = self.obj.tags
                data.description = self.obj.description

        except AttributeError:
            warning(
                f"'tags' and 'description' can't be attached to {type(data)} object."
            )
        return data

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


@serializable()
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
        # TODO: remove delete_obj flag if everthing works fine.
        self.id_at_location = id_at_location
        self.delete_obj = delete_obj

        # the logger needs self.id_at_location to be set already - so we call this later
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:
        try:
            try:
                storable_object = node.store.get(self.id_at_location, proxy_only=True)
            except Exception as e:
                log = (
                    f"Unable to Get Object with ID {self.id_at_location} from store. "
                    + f"Possible dangling Pointer. {e}"
                )

                traceback_and_raise(Exception(log))

            # if you are not the root user check if your verify_key has read_permission
            if (
                verify_key != node.root_verify_key
                and verify_key not in storable_object.read_permissions
            ):
                log = (
                    f"You do not have permission to .get() Object with ID: {self.id_at_location} on node {node.name} "
                    + "Please submit a request."
                )
                traceback_and_raise(AuthorizationException(log))

            obj = validate_type(
                storable_object.clean_copy(settings=node.settings), StorableObject
            )

            if obj.is_proxy:
                obj.data.generate_presigned_url(settings=node.settings, public_url=True)

            msg = GetObjectResponseMessage(obj=obj, address=self.reply_to, msg_id=None)

            debug(
                f"Returning Object with ID: {self.id_at_location} {type(storable_object.data)}"
            )
            return msg
        except Exception as e:
            traceback_and_raise(e)
        raise Exception(f"Unable to execute action with {type(self)}")

    @property
    def pprint(self) -> str:
        return f"GetObjectAction({self.id_at_location})"

    def _object2proto(self) -> GetObjectAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectWithID_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetObjectAction_PB(
            id_at_location=serialize(self.id_at_location, to_proto=True),
            msg_id=serialize(self.id, to_proto=True),
            address=serialize(self.address, to_proto=True),
            reply_to=serialize(self.reply_to, to_proto=True),
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
            id_at_location=deserialize(blob=proto.id_at_location),
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            reply_to=deserialize(blob=proto.reply_to),
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
