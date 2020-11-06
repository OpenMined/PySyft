# stdlib
import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ...decorators import syft_decorator
from ...proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from ...util import get_fully_qualified_name
from ...util import key_emoji
from ..common.group import All
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import Serializable
from ..common.storeable_object import AbstractStorableObject
from ..common.uid import UID


class StorableObject(AbstractStorableObject):
    """
    StorableObject is a wrapper over some Serializable objects, which we want to keep in an
    ObjectStore. The Serializable objects that we want to store have to be backed up in syft-proto
    in the StorableObject protobuffer, where you can find more details on how to add new types to be
    serialized.

    This object is frozen, you cannot change one in place.

    Arguments:
        id (UID): the id at which to store the data.
        data (Serializable): A serializable object.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.

    Attributes:
        id (UID): the id at which to store the data.
        data (Serializable): A serializable object.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.

    """

    __slots__ = ["id", "data", "description", "tags"]

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        id: UID,
        data: object,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        read_permissions: Optional[Dict[VerifyKey, Optional[UID]]] = {},
        search_permissions: Optional[Dict[Union[VerifyKey, All], Optional[UID]]] = {},
    ):
        self.id = id
        self.data = data
        self.description = (
            getattr(data, "description", "") if description is None else description
        )
        self.tags = getattr(data, "tags", []) if tags is None else tags

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to call .get() and download this object.
        self.read_permissions = read_permissions

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to know that the tensor exists (via search or other means)
        self.search_permissions = search_permissions

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> StorableObject_PB:
        proto = StorableObject_PB()
        id = self.id.serialize()
        proto.id.CopyFrom(id)
        proto.obj_type = get_fully_qualified_name(obj=self.data)
        data = self._data_object2proto()
        proto.data.Pack(data)

        proto.description = self.description

        for tag in self.tags:
            proto.tags.append(tag)

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: StorableObject_PB) -> object:
        id = _deserialize(blob=proto.id)

        fqn = proto.obj_type

        # INSANE HACK INCOMING DUE TO NOT PROPERLY USING WRAPPERS, TODO for Tudor to fix it,
        # use it to unblock Jay

        if fqn == "torch.Tensor":
            # syft relative
            from ...lib.torch.uppercase_tensor import TorchTensorWrapper
            from ...proto.lib.torch.tensor_pb2 import TensorProto

            tensor_data = TensorProto()
            proto.data.Unpack(tensor_data)
            return TorchTensorWrapper._data_proto2object(proto=tensor_data)

        if fqn == "torch.nn.parameter.Parameter":
            # syft relative
            from ...lib.torch.parameter import PyTorchParameterWrapper
            from ...proto.lib.torch.parameter_pb2 import ParameterProto

            tensor_data = ParameterProto()
            proto.data.Unpack(tensor_data)
            return PyTorchParameterWrapper._data_proto2object(proto=tensor_data)

        parts = fqn.split(".")
        klass_name = parts.pop()
        obj_type = getattr(sys.modules[".".join(parts)], klass_name)

        # Step 4: get the protobuf type we deserialize for .data
        schematic_type = obj_type.get_protobuf_schema()

        # Step 4: Deserialize data from protobuf
        data = schematic_type()
        descriptor = getattr(schematic_type, "DESCRIPTOR", None)
        if descriptor is not None and proto.data.Is(descriptor):
            proto.data.Unpack(data)
        # if issubclass(type(target_type), Serializable):
        data = obj_type._proto2object(proto=data)

        # Step 5: get the description from proto
        description = proto.description

        # Step 6: get the tags from proto of they exist
        tags = None
        if proto.tags:
            tags = list(proto.tags)

        result = StorableObject.construct_new_object(
            id=id, data=data, tags=tags, description=description
        )

        # just a backup
        result.tags = tags
        result.description = description

        # this is quite insane
        return result.data

    def _data_object2proto(self) -> Message:
        return self.data.serialize()  # type: ignore

    @staticmethod
    def _data_proto2object(proto: Message) -> Serializable:
        return _deserialize(blob=proto)

    @staticmethod
    def get_data_protobuf_schema() -> Optional[Type]:
        return None

    @staticmethod
    def construct_new_object(
        id: UID,
        data: "StorableObject",
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> "StorableObject":
        return StorableObject(id=id, data=data, description=description, tags=tags)

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
        return StorableObject_PB

    def __repr__(self) -> str:
        return (
            "<Storable:"
            + self.data.__repr__().replace("\n", "").replace("  ", " ")
            + ">"
        )

    @property
    def icon(self) -> str:
        return "🗂️"

    @property
    def pprint(self) -> str:
        output = f"{self.icon} ({self.class_name}) ("
        if hasattr(self.data, "pprint"):
            output += self.data.pprint  # type: ignore
        elif self.data is not None:
            output += self.data.__repr__()
        else:
            output += "(Key Only)"
        if self.description is not None and len(self.description) > 0:
            output += f" desc: {self.description}"
        if self.tags is not None and len(self.tags) > 0:
            output += f" tags: {self.tags}"
        if self.read_permissions is not None and len(self.read_permissions.keys()) > 0:
            output += (
                " can_read: "
                + f"{[key_emoji(key=key) for key in self.read_permissions.keys()]}"
            )

        if (
            self.search_permissions is not None
            and len(self.search_permissions.keys()) > 0
        ):
            output += (
                " can_search: "
                + f"{[key_emoji(key=key) for key in self.search_permissions.keys()]}"
            )

        output += ")"
        return output

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)
