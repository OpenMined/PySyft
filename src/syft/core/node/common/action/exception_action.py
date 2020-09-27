# stdlib
import sys
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.exception_action_pb2 import (
    ExceptionMessage as ExceptionMessage_PB,
)
from .....util import get_fully_qualified_name
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address


class UnknownPrivateException(Exception):
    pass


@final
class ExceptionMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        msg_id_causing_exception: UID,
        exception_type: Type,
        exception_msg: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.msg_id_causing_exception = msg_id_causing_exception
        self.exception_type = exception_type
        self.exception_msg = exception_msg

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> ExceptionMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ExceptionMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        # convert exception into fully qualified class path
        fqn = get_fully_qualified_name(obj=self.exception_type)
        module_parts = fqn.split(".")
        _ = module_parts.pop()  # remove incorrect .type ending
        module_parts.append(self.exception_type.__name__)
        fqn = ".".join(module_parts)

        return ExceptionMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            msg_id_causing_exception=self.msg_id_causing_exception.serialize(),
            exception_type=fqn,
            exception_msg=self.exception_msg,
        )

    @staticmethod
    def _proto2object(proto: ExceptionMessage_PB) -> "ExceptionMessage":
        """Creates a ExceptionMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ExceptionMessage
        :rtype: ExceptionMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        # converting fqn of exception class back to real Exception
        module_parts = proto.exception_type.split(".")
        klass = module_parts.pop()
        exception_type = getattr(sys.modules[".".join(module_parts)], klass)

        return ExceptionMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            msg_id_causing_exception=_deserialize(blob=proto.msg_id_causing_exception),
            exception_type=exception_type,
            exception_msg=proto.exception_msg,
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

        return ExceptionMessage_PB
