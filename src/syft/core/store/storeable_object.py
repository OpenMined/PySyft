import pydoc
from typing import List, Optional
from ...decorators import syft_decorator
from ...proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.deserialize import _deserialize
from ..common.uid import UID
from google.protobuf.message import Message
from ...util import get_fully_qualified_name


class StorableObject(Serializable):
    """
    StorableObject is a wrapper over some Serializable objects, which we want to keep in an
    ObjectStore. The Serializable objects that we want to store have to be backed up in syft-proto
    in the StorableObject protobuffer, where you can find more details on how to add new types to be
    serialized.

    This object is frozen, you cannot change one in place.

    Arguments:
        key (UID): the key at which to store the data.
        data (Serializable): A serializable object.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.

    Attributes:
        key (UID): the key at which to store the data.
        data (Serializable): A serializable object.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.

    """

    __slots__ = ["key", "data", "description", "tags"]

    # TODO: remove this flag if commenting it out doesn't break anything
    # protobuf_type = StorableObject_PB

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        key: UID,
        data: object,
        description: Optional[str],
        tags: Optional[List[str]],
    ):
        self.key = key
        self.data = data
        self.description = description
        self.tags = tags

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> StorableObject_PB:

        proto = StorableObject_PB()

        # Step 1: Serialize the key to protobuf and copy into protobuf
        key = self.key.serialize()
        proto.key.CopyFrom(key)
        #
        # # Step 2: save the type of object we're about to serialize
        # proto.schematic_qualname = get_fully_qualified_name(obj=self.data)
        # print("Underlying Object Type:" + str(proto.schematic_qualname))

        # Step 3: Save the type of wrapper to use to deserialize
        proto.obj_type = get_fully_qualified_name(obj=self)
        print("Underlying Wrapper Type:" + str(proto.obj_type))

        # Step 4: Serialize data to protobuf and pack into proto
        data = self._data_object2proto()
        proto.data.Pack(data)

        # Step 5: save the description into proto
        proto.description = self.description

        # Step 6: save tags into proto if they exist
        if self.tags is not None:
            for tag in self.tags:
                proto.tags.append(tag)

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: StorableObject_PB) -> object:
        # Step 1: deserialize the ID
        key = _deserialize(blob=proto.key)

        # Step 2: get the type of object we're about to serialize
        # data_type = pydoc.locate(proto.schematic_qualname)
        # # from syft.proto.lib.numpy.tensor_pb2 import TensorProto
        #
        # schematic_type = data_type

        # Step 3: get the type of wrapper to use to deserialize
        obj_type: StorableObject = pydoc.locate(proto.obj_type)  # type: ignore
        target_type = obj_type

        # Step 4: get the protobuf type we deserialize for .data
        schematic_type = obj_type.get_data_protobuf_schema()

        # Step 4: Deserialize data from protobuf
        if schematic_type is not None and callable(schematic_type):
            data = schematic_type()
            descriptor = getattr(schematic_type, "DESCRIPTOR", None)
            if descriptor is not None and proto.data.Is(descriptor):
                proto.data.Unpack(data)
            # if issubclass(type(target_type), Serializable):
            data = target_type._data_proto2object(proto=data)
        else:
            data = None
        # Step 5: get the description from proto
        description = proto.description

        # Step 6: get the tags from proto of they exist
        tags = None
        if proto.tags:
            tags = list(proto.tags)

        return target_type.construct_new_object(
            id=key, data=data, tags=tags, description=description
        )

    def _data_object2proto(self) -> Message:
        return self.data.serialize()  # type: ignore

    @staticmethod
    def _data_proto2object(proto: Message) -> int:
        return _deserialize(blob=proto)

    @staticmethod
    def get_data_protobuf_schema() -> None:
        return None

    @staticmethod
    def construct_new_object(id, data, tags, description):
        return StorableObject(key=id, data=data, description=description, tags=tags)

    @staticmethod
    def get_protobuf_schema() -> type:
        return StorableObject_PB
