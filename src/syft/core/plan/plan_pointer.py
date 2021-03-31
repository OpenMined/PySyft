# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# syft relative
from ...proto.core.plan.plan_pointer_pb2 import PlanPointer as PlanPointer_PB
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import bind_protobuf
from ..pointer.pointer import Pointer
from ..common.uid import UID


@bind_protobuf
class PlanPointer():

    def __init__(
        self,
        pointer: Pointer,
        seq_id: str,
        path_and_name: str = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        if isinstance(pointer, Pointer):
            self.seq_id = seq_id
            self.tags = pointer.tags
            self.description = pointer.description
            self.object_type = pointer.object_type
            self.path_and_name = pointer.path_and_name
            self.pointer_name = type(pointer).__name__
        else:
            self.tags = tags
            self.description = description
            self.object_type = object_type
            self.path_and_name = path_and_name
            self.pointer_name = pointer
            self.seq_id = seq_id

    def get_pointer_obj(
        self,
        client: Any
    ) -> Pointer:
        points_to_type = sy.lib_ast.query(self.path_and_name)
        pointer_type = getattr(points_to_type, self.pointer_name)

        return pointer_type(
            client=client,
            tags=self.tags,
            description=self.description,
            object_type=self.object_type,
        )

    def _object2proto(self) -> PlanPointer_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: Pointer_PB

        .. note::
            This method is purely an internal method. Please use sy.serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return PlanPointer_PB(
            points_to_object_with_path=self.path_and_name,
            pointer_name=self.pointer_name,
            object_type=self.object_type,
            tags=self.tags,
            description=self.description,
            seq_id=self.seq_id
        )

    @staticmethod
    def _proto2object(proto: PlanPointer_PB) -> "Pointer":
        """Creates a Pointer from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Pointer
        :rtype: Pointer

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return PlanPointer(
            pointer=proto.pointer_name,
            path_and_name=proto.points_to_object_with_path,
            object_type=proto.object_type,
            tags=proto.tags,
            description=proto.description,
            seq_id=proto.seq_id,
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

        return PlanPointer_PB
