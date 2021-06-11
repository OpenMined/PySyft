# stdlib
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syft_proto.execution.v1.placeholder_pb2 import Placeholder as PlaceholderPB
import torch as th

# syft relative
from ...core.common.object import Serializable
from .common import get_protobuf_id
from .common import set_protobuf_id
from .placeholder_id import PlaceholderId


class PlaceHolder(Serializable):
    def __init__(
        self,
        id: Union[int, str],
        tags: Set = set(),
        description: Optional[str] = None,
        shape: Optional[Union[th.Size, Tuple[int]]] = None,
        expected_dtype: Optional[th.dtype] = None,
    ):
        """A PlaceHolder acts as a tensor but does nothing special. It can get
        "instantiated" when a real tensor is appended as a child attribute. It
        will send forward all the commands it receives to its child tensor.

        When you send a PlaceHolder, you don't sent the instantiated tensors.

        Args:
            id: An optional string or integer id of the PlaceHolder.
        """
        super().__init__()

        self.id = PlaceholderId(id)
        self.tags = tags
        self.description = description
        self.expected_shape: Optional[Tuple] = (
            tuple(shape) if shape is not None else None
        )
        self.expected_dtype = expected_dtype
        self.child: Optional[Union[th.Tensor, th.nn.Parameter, "PlaceHolder"]] = None

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
        return PlaceholderPB

    def _object2proto(self) -> PlaceholderPB:
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

        protobuf_placeholder = PlaceholderPB()
        set_protobuf_id(protobuf_placeholder.id, self.id.value)
        protobuf_placeholder.tags.extend(self.tags)

        if self.description:
            protobuf_placeholder.description = self.description

        if self.expected_shape:
            protobuf_placeholder.expected_shape.dims.extend(self.expected_shape)

        return protobuf_placeholder

    @staticmethod
    def _proto2object(proto: PlaceholderPB) -> "PlaceHolder":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Plan
        :rtype: Plan

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        tensor_id = get_protobuf_id(proto.id)
        tags = set(proto.tags) if proto.tags else set()

        description = None
        if bool(proto.description):
            description = proto.description

        expected_shape = tuple(proto.expected_shape.dims) or None

        return PlaceHolder(
            id=tensor_id, tags=tags, description=description, shape=expected_shape
        )

    def instantiate(self, tensor: Union[th.Tensor, th.nn.Parameter]) -> "PlaceHolder":
        """
        Add a tensor as a child attribute. All operations on the placeholder will be also
        executed on this child tensor.

        We remove Placeholders if is there are any.
        """
        if isinstance(tensor, PlaceHolder):
            self.child = tensor.child
        else:
            self.child = tensor

        shape = getattr(self.child, "shape", None)
        if shape is not None:
            self.expected_shape = tuple(shape)

        dtype = getattr(self.child, "dtype", None)
        if dtype is not None:
            self.expected_dtype = dtype

        return self
