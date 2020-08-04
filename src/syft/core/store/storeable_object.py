from __future__ import annotations

import pydoc
from typing import List, Optional
from ...decorators import syft_decorator
from ...proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.deserialize import _deserialize
from ..common.uid import UID


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

    protobuf_type = StorableObject_PB

    @syft_decorator(typechecking=True)
    def __init__(self, key: UID, data: Serializable, description: Optional[str],
                 tags: Optional[List[str]]):
        super().__init__(as_wrapper=False)
        self.key = key
        self.data = data
        self.description = description
        self.tags = tags

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> StorableObject_PB:
        key = self.key.serialize()
        data = self.data.serialize()
        proto = StorableObject_PB()
        proto.key.CopyFrom(key)
        proto.schematic_qualname = "syft." + type(data).__module__ + "." + type(data).__name__
        proto.data.Pack(data)
        proto.obj_type = type(self).__module__ + "." + type(self).__name__
        proto.description=self.description
        for tag in self.tags:
            proto.tags.append(tag)
        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: StorableObject_PB) -> "StorableObject":
        key = _deserialize(proto.key)
        schematic_type = pydoc.locate(proto.schematic_qualname)
        target_type = pydoc.locate(proto.obj_type)
        schematic = schematic_type()
        if proto.data.Is(schematic_type.DESCRIPTOR):
            proto.data.Unpack(schematic)
        data = target_type._proto2object(proto=schematic)
        tags = None
        if proto.tags:
            tags = list(proto.tags)

        return StorableObject(
            key=key, data=data, description=proto.description, tags=tags
        )
