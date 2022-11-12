# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...proto.core.store.dataset_pb2 import Dataset as Dataset_PB
from ...util import get_fully_qualified_name
from ..common.serde.deserialize import _deserialize as deserialize  # noqa: F401
from ..common.serde.serializable import serializable
from ..common.serde.serialize import _serialize as serialize  # noqa: F401
from ..common.uid import UID
from .storeable_object import StorableObject


@serializable()
class Dataset:
    """
    Dataset is a wrapper over a collection of Serializable objects.

    Arguments:
        id (UID): the id at which to store the data.
        data (List[Serializable]): A list of serializable objects.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.
        TODO: add docs about read_permission and search_permission

    Attributes:
        id (UID): the id at which to store the data.
        data (List[Serializable]): A list of serializable objects.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.

    """

    def __init__(
        self,
        id: UID,
        data: List[StorableObject],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        read_permissions: Optional[dict] = None,
        search_permissions: Optional[dict] = None,
    ):
        self.id = id
        self.data = data
        self._description: str = description if description else ""
        self._tags: List[str] = tags if tags else []

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to call .get() and download this object.
        self.read_permissions = read_permissions if read_permissions else {}

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to know that the tensor exists (via search or other means)
        self.search_permissions: dict = search_permissions if search_permissions else {}

    @property
    def tags(self) -> Optional[List[str]]:
        return self._tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        self._tags = value if value else []

    @property
    def description(self) -> Optional[str]:
        return self._description

    @description.setter
    def description(self, description: Optional[str]) -> None:
        self._description = description if description else ""

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    def __contains__(self, _id: UID) -> bool:
        return _id in [el.id for el in self.data]

    def keys(self) -> List[UID]:
        return [el.id for el in self.data]

    def __getitem__(self, _id: UID) -> List[StorableObject]:
        return [el for el in self.data if el.id == _id]

    def __delitem__(self, _id: UID) -> None:
        self.data = [el for el in self.data if el.id != _id]

    def _object2proto(self) -> Dataset_PB:
        # relative
        from ...lib.python.dict import Dict

        proto = Dataset_PB()

        # Step 1: Serialize the id to protobuf and copy into protobuf
        id = serialize(self.id)
        proto.id.CopyFrom(id)

        # Step 2: Save the type of wrapper to use to deserialize
        proto.obj_type = get_fully_qualified_name(obj=self)

        # Step 3: Serialize data to protobuf and pack into proto
        if hasattr(self, "data"):
            if self.data is not None:
                for _d in self.data:
                    proto_storable = _d._object2proto()
                    proto.data.append(proto_storable)

        if hasattr(self, "description"):
            # Step 4: save the description into proto
            proto.description = self.description

        # QUESTION: Which one do we want, self.data.tags or self.tags or both???
        if hasattr(self, "tags"):
            # Step 5: save tags into proto if they exist
            if self.tags is not None:
                for tag in self.tags:
                    proto.tags.append(tag)

        # Step 6: save read permissions
        if len(self.read_permissions.keys()) > 0:
            permission_data = Dict()
            for k, v in self.read_permissions.items():
                permission_data[k] = v
            proto.read_permissions = serialize(permission_data, to_bytes=True)

        # Step 7: save search permissions
        if len(self.search_permissions.keys()) > 0:
            permission_data = Dict()
            for k, v in self.search_permissions.items():
                permission_data[k] = v
            proto.search_permissions = serialize(permission_data, to_bytes=True)

        return proto

    @staticmethod
    def _proto2object(proto: Dataset_PB) -> "Dataset":

        # Step 1: deserialize the ID
        id = deserialize(blob=proto.id)

        if not isinstance(id, UID):
            raise ValueError("TODO")

        # Step 2: Deserialize data from protobuf
        data = list(proto.data) if proto.data else []
        data = [StorableObject._proto2object(proto=d) for d in data]

        # Step 3: get the description from proto
        description = proto.description if proto.description else ""

        # Step 4: get the tags from proto of they exist
        tags = list(proto.tags) if proto.tags else []

        result = Dataset(id=id, data=data, description=description, tags=tags)
        return result

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
        return Dataset_PB
