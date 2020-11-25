# stdlib
import pydoc
from typing import List
from typing import Optional

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger

# syft absolute
import syft as sy

# syft relative
from ...decorators import syft_decorator
from ...proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from ...util import get_fully_qualified_name
from ...util import key_emoji
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
        TODO: add docs about read_permission and search_permission

    Attributes:
        id (UID): the id at which to store the data.
        data (Serializable): A serializable object.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.

    """

    __slots__ = ["id", "data", "_description", "_tags"]

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        id: UID,
        data: object,
        description: str = "",
        tags: Optional[List[str]] = None,
        read_permissions: Optional[dict] = None,
        search_permissions: Optional[dict] = None,
    ):
        self.id = id
        self.data = data
        self._description: str = description
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

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> StorableObject_PB:
        proto = StorableObject_PB()

        # Step 1: Serialize the id to protobuf and copy into protobuf
        id = self.id.serialize()
        proto.id.CopyFrom(id)

        # Step 2: Save the type of wrapper to use to deserialize
        proto.obj_type = get_fully_qualified_name(obj=self)

        # Step 3: Serialize data to protobuf and pack into proto
        data = self._data_object2proto()

        proto.data.Pack(data)

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
            permission_data = sy.lib.python.Dict()
            for k, v in self.read_permissions.items():
                permission_data[k] = v
            proto.read_permissions = permission_data.serialize(to_bytes=True)

        # Step 7: save search permissions
        if len(self.search_permissions.keys()) > 0:
            permission_data = sy.lib.python.Dict()
            for k, v in self.search_permissions.items():
                permission_data[k] = v
            proto.search_permissions = permission_data.serialize(to_bytes=True)

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: StorableObject_PB) -> object:

        # Step 1: deserialize the ID
        id = _deserialize(blob=proto.id)

        # TODO: FIX THIS SECURITY BUG!!! WE CANNOT USE
        #  PYDOC.LOCATE!!!
        # Step 2: get the type of wrapper to use to deserialize
        obj_type: StorableObject = pydoc.locate(proto.obj_type)  # type: ignore

        # Step 3: get the protobuf type we deserialize for .data
        schematic_type = obj_type.get_data_protobuf_schema()

        # Step 4: Deserialize data from protobuf
        data = None
        if callable(schematic_type):
            data = schematic_type()
            descriptor = getattr(schematic_type, "DESCRIPTOR", None)
            if descriptor is not None and proto.data.Is(descriptor):
                proto.data.Unpack(data)
            data = obj_type._data_proto2object(proto=data)

        # Step 5: get the description from proto
        description = proto.description if proto.description else ""

        # Step 6: get the tags from proto of they exist
        tags = list(proto.tags) if proto.tags else []

        result = obj_type.construct_new_object(
            id=id, data=data, tags=tags, description=description
        )

        # just a backup
        try:
            result.tags = tags
            result.description = description

            # default to empty
            result.read_permissions = {}
            result.search_permissions = {}

            # Step 7: get the read permissions
            if proto.read_permissions is not None and len(proto.read_permissions) > 0:
                result.read_permissions = _deserialize(
                    blob=proto.read_permissions, from_bytes=True
                )

            # Step 8: get the search permissions
            if (
                proto.search_permissions is not None
                and len(proto.search_permissions) > 0
            ):
                result.search_permissions = _deserialize(
                    blob=proto.search_permissions, from_bytes=True
                )
        except Exception as e:
            # torch.return_types.* namedtuple cant setattr
            log = f"StorableObject {type(obj_type)} cant set attributes {e}"
            logger.error(log)

        return result

    def _data_object2proto(self) -> Message:
        return self.data.serialize()  # type: ignore

    @staticmethod
    def _data_proto2object(proto: Message) -> Serializable:
        return _deserialize(blob=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return StorableObject_PB

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
            "<Storable: "
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
        if len(self._description) > 0:
            output += f" desc: {self.description}"
        if len(self._tags) > 0:
            output += f" tags: {self.tags}"
        if len(self.read_permissions.keys()) > 0:
            output += (
                " can_read: "
                + f"{[key_emoji(key=key) for key in self.read_permissions.keys()]}"
            )

        if len(self.search_permissions.keys()) > 0:
            output += (
                " can_search: "
                + f"{[key_emoji(key=key) for key in self.search_permissions.keys()]}"
            )

        output += ")"
        return output

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)
