# stdlib
from typing import List

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft_proto.execution.v1.state_tensor_pb2 import StateTensor as StateTensorPB

# syft relative
from ...core.common.object import Serializable
from ...core.common.serde.serialize import _serialize as serialize
from .common import deserialize_tensor
from .common import serialize_tensor
from .placeholder import PlaceHolder


class State(Serializable):
    """The State is a Plan attribute and is used to send tensors along functions.

    It references Plan tensor or parameters attributes using their name, and make
    sure they are provided to remote workers who are sent the Plan.
    """

    def __init__(self, state_placeholders: List[PlaceHolder] = []) -> None:
        self.state_placeholders = state_placeholders

    def tensors(self) -> List:
        """
        Fetch and return all the state elements.
        """
        return [placeholder.child for placeholder in self.state_placeholders]

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
        return StatePB

    def _object2proto(self) -> StatePB:
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
        proto = StatePB()
        protobuf_placeholders = [
            serialize(placeholder) for placeholder in self.state_placeholders
        ]
        proto.placeholders.extend(protobuf_placeholders)

        state_tensors = []
        for tensor in self.tensors():
            state_tensor = StateTensorPB()
            state_tensor.torch_tensor.CopyFrom(serialize_tensor(tensor))
            state_tensors.append(state_tensor)

        proto.tensors.extend(state_tensors)
        return proto

    @staticmethod
    def _proto2object(proto: StatePB) -> "State":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Plan
        :rtype: Plan

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        state_placeholders = proto.placeholders
        state_elements = proto.tensors

        state_placeholders = [
            PlaceHolder._proto2object(placeholder) for placeholder in proto.placeholders
        ]

        state_elements = []
        for protobuf_tensor in proto.tensors:
            tensor = getattr(protobuf_tensor, protobuf_tensor.WhichOneof("tensor"))
            state_elements.append(deserialize_tensor(tensor))

        for state_placeholder, state_element in zip(state_placeholders, state_elements):
            state_placeholder.instantiate(state_element)

        state = State(state_placeholders)
        return state
