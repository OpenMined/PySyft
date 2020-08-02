from typing import List, Optional, Union
from dataclasses import dataclass

from ..common.uid import UID
from ..common.serializable import Serializable
from ...proto.core.store.storable_object_pb2 import StorableObject as StorableObject_PB
from ..common.serializable import deserialize

@dataclass(frozen=True)
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

    key: UID
    data: Serializable
    description: Optional[str]
    tags: Optional[List[str]]

    def _object2proto(self) -> "StorableObject_PB":
        key = self.key.serialize()
        data = self.data.serialize()
        return StorableObject_PB(key=key, data=data, description=self.description, tags=self.tags)

    @staticmethod
    def _proto2object(proto: StorableObject_PB) -> "StorableObject":
        key = deserialize(proto.key)
        data = deserialize(proto.data)
        tags = None
        if proto.tags:
            tags = list(proto.tags)

        return StorableObject(key=key, data=data, description=proto.description, tags=tags)
