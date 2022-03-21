# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from pydantic import BaseSettings

# syft absolute
import syft as sy

# relative
from ...logger import traceback_and_raise
from ...proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from ...util import get_fully_qualified_name
from ...util import index_syft_by_module_name
from ...util import key_emoji
from ..common.serde.deserialize import CapnpMagicBytesNotFound
from ..common.serde.deserialize import _deserialize
from ..common.serde.deserialize import deserialize_capnp
from ..common.serde.serializable import serializable
from ..common.serde.serialize import _serialize
from ..common.storeable_object import AbstractStorableObject
from ..common.uid import UID
from .proxy_dataset import ProxyDataset


@serializable()
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

    __slots__ = ["id", "_data", "_description", "_tags"]

    def __init__(
        self,
        id: UID,
        data: object,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        read_permissions: Optional[dict] = None,
        search_permissions: Optional[dict] = None,
        write_permissions: Optional[dict] = None,
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
        self.write_permissions: dict = write_permissions if write_permissions else {}

    @property
    def object_type(self) -> str:
        return str(type(self.data))

    @property
    def object_qualname(self) -> str:
        return get_fully_qualified_name(self.data)

    # Why define data as a property?
    # For C type/class objects as data.
    # We need to use it's wrapper type very often inside StorableObject, so we set _data
    # attribute as it's wrapper object. But we still want to give a straight API to users,
    # so we return the initial C type object when user call obj.data.
    # For python class objects as data. data and _data are the same thing.
    @property  # type: ignore
    def data(self) -> Any:  # type: ignore
        if type(self._data).__name__.endswith("Wrapper"):
            return self._data.obj
        else:
            return self._data

    @data.setter
    def data(self, value: Any) -> Any:
        if hasattr(value, "_sy_serializable_wrapper_type"):
            self._data = value._sy_serializable_wrapper_type(value=value)
        else:
            self._data = value

    @property
    def is_proxy(self) -> bool:
        return isinstance(self._data, ProxyDataset)

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

    def _object2proto(self) -> StorableObject_PB:
        # relative
        from ...lib.python.bytes import Bytes

        proto = StorableObject_PB()

        # Step 1: Serialize the id to protobuf and copy into protobuf
        id = sy.serialize(self.id)
        proto.id.CopyFrom(id)

        # Step 2: Save the type of wrapper to use to deserialize
        proto.data_type = get_fully_qualified_name(obj=self._data)

        # Step 3: Serialize data to protobuf and pack into proto
        if hasattr(self._data, "_object2bytes"):
            data = Bytes(self._data._object2bytes())._object2proto()
            proto.data_type = get_fully_qualified_name(obj=Bytes())
        elif hasattr(self._data, "_object2proto"):
            data = self._data._object2proto()
        else:
            # @Tudor this needs fixing during the serde refactor
            # we should probably just support the native type names as lookups for serde
            data = _serialize(self._data, to_proto=True)

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
        if len(self.read_permissions) > 0:
            permission_data = sy.lib.python.Dict()
            for k, v in self.read_permissions.items():
                permission_data[k] = v
            proto.read_permissions = sy.serialize(permission_data, to_bytes=True)

        # Step 7: save search permissions
        if len(self.search_permissions.keys()) > 0:
            permission_data = sy.lib.python.Dict()
            for k, v in self.search_permissions.items():
                permission_data[k] = v
            proto.search_permissions = sy.serialize(permission_data, to_bytes=True)

        # Step 8: save write permissions
        if len(self.write_permissions.keys()) > 0:
            permission_data = sy.lib.python.Dict()
            for k, v in self.write_permissions.items():
                permission_data[k] = v
            proto.write_permissions = sy.serialize(permission_data, to_bytes=True)

        return proto

    @staticmethod
    def _proto2object(proto: StorableObject_PB) -> "StorableObject":
        # relative
        from ...lib.python.bytes import Bytes

        # Step 1: deserialize the ID
        id = _deserialize(blob=proto.id)

        if not isinstance(id, UID):
            traceback_and_raise(ValueError("TODO"))

        # Step 2: get the type of wrapper to use to deserialize
        data_type = index_syft_by_module_name(fully_qualified_name=proto.data_type)

        # Step 3: get the protobuf type we deserialize for .data
        schematic_type = data_type.get_protobuf_schema()  # type: ignore

        # Step 4: Deserialize data from protobuf
        data = None
        if callable(schematic_type):
            data = schematic_type()
            descriptor = getattr(schematic_type, "DESCRIPTOR", None)
            if descriptor is not None and proto.data.Is(descriptor):
                proto.data.Unpack(data)
            data = data_type._proto2object(proto=data)  # type: ignore

        if isinstance(data, (Bytes, bytes)):
            try:
                data = deserialize_capnp(buf=data)
            except CapnpMagicBytesNotFound:
                data = sy.deserialize(data, from_bytes=True)
            except Exception as e:
                traceback_and_raise(f"Failed to deserialize Bytes with capnp. {e}")

        # Step 5: get the description from proto
        description = proto.description if proto.description else ""

        # Step 6: get the tags from proto of they exist
        tags = list(proto.tags) if proto.tags else []

        # Step 7: get the read permissions
        read_permissions = None
        if proto.read_permissions is not None and len(proto.read_permissions) > 0:
            read_permissions = _deserialize(
                blob=proto.read_permissions, from_bytes=True
            )

        # Step 8: get the search permissions
        search_permissions = None
        if proto.search_permissions is not None and len(proto.search_permissions) > 0:
            search_permissions = _deserialize(
                blob=proto.search_permissions, from_bytes=True
            )

        # Step 9: get the write permissions
        write_permissions = None
        if proto.write_permissions is not None and len(proto.write_permissions) > 0:
            write_permissions = _deserialize(
                blob=proto.write_permissions, from_bytes=True
            )

        result = StorableObject(
            id=id,
            data=data,
            description=description,
            tags=tags,
            read_permissions=read_permissions,
            search_permissions=search_permissions,
            write_permissions=write_permissions,
        )

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
            output += self.data.pprint
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
        if len(self.write_permissions.keys()) > 0:
            output += (
                " can_write: "
                + f"{[key_emoji(key=key) for key in self.write_permissions.keys()]}"
            )

        output += ")"
        return output

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    def clean_copy(self, settings: BaseSettings) -> "StorableObject":
        """
        This method return a copy of self, but clean up the search_permissions and
        read_permissions attributes.
        """
        return StorableObject(
            id=self.id, data=self.data, tags=self.tags, description=self.description
        )
