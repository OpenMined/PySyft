from typing import List, Optional
from dataclasses import dataclass

from ..common.uid import UID
from ..common.serializable import Serializable


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

    @staticmethod
    def to_protobuf(self):
        schema = StorableObjectPB()

        raise NotImplementedError

    @staticmethod
    def from_protobuf(proto):
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema():
        raise NotImplementedError

    @staticmethod
    def get_wrapped_type():
        raise NotImplementedError
