# stdlib
import pydoc
from typing import Dict as DictType
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from google.protobuf.empty_pb2 import Empty as Empty_PB
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# syft relative
from ...decorators import syft_decorator
from ...proto.core.auth.signed_message_pb2 import VerifyAll as VerifyAllWrapper_PB
from ...proto.core.auth.signed_message_pb2 import VerifyKey as VerifyKeyWrapper_PB
from ...proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from ...util import aggressive_set_attr
from ...util import get_fully_qualified_name
from ...util import key_emoji
from ..common.group import VerifyAll
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
        description: Optional[str] = "",
        tags: Optional[List[str]] = [],
        read_permissions: Optional[DictType[VerifyKey, Optional[UID]]] = {},
        search_permissions: Optional[
            DictType[Union[VerifyKey, VerifyAll], Optional[UID]]
        ] = {},
    ):
        self.id = id
        self.data = data
        self.description = description
        self.tags = tags

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

        # Step 1: Serialize the id to protobuf and copy into protobuf
        id = self.id.serialize()
        proto.id.CopyFrom(id)

        # Step 2: Save the type of wrapper to use to deserialize
        proto.obj_type = get_fully_qualified_name(obj=self)

        # Step 3: Serialize data to protobuf and pack into proto
        data = self._data_object2proto()

        proto.data.Pack(data)

        if hasattr(self.data, "description"):
            # Step 4: save the description into proto
            proto.description = self.data.description  # type: ignore

        # QUESTION: Which one do we want, self.data.tags or self.tags or both???
        if hasattr(self.data, "tags"):
            # Step 5: save tags into proto if they exist
            if self.data.tags is not None and self.tags is not None:  # type: ignore
                for tag in self.tags:
                    proto.tags.append(tag)

        # Step 6: save read permissions
        if self.read_permissions is not None and len(self.read_permissions.keys()) > 0:
            permission_data = sy.lib.python.Dict()
            for k, v in self.read_permissions.items():
                permission_data[k] = v
            proto.read_permissions = permission_data.serialize(to_bytes=True)

        # Step 7: save search permissions
        if (
            self.search_permissions is not None
            and len(self.search_permissions.keys()) > 0
        ):
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
        if schematic_type is not None and callable(schematic_type):
            data = schematic_type()
            descriptor = getattr(schematic_type, "DESCRIPTOR", None)
            if descriptor is not None and proto.data.Is(descriptor):
                proto.data.Unpack(data)
            data = obj_type._data_proto2object(proto=data)
        else:
            data = None

        # Step 5: get the description from proto
        description = proto.description

        # Step 6: get the tags from proto of they exist
        tags = None
        if proto.tags:
            tags = list(proto.tags)

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


class VerifyKeyWrapper(StorableObject):
    def __init__(self, value: VerifyKey):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> VerifyKeyWrapper_PB:
        return VerifyKeyWrapper_PB(verify_key=bytes(self.value))

    @staticmethod
    def _data_proto2object(proto: VerifyKeyWrapper_PB) -> VerifyKey:
        return VerifyKey(proto.verify_key)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return VerifyKeyWrapper_PB

    @staticmethod
    def get_wrapped_type() -> Type:
        return VerifyKey

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=VerifyKey, name="serializable_wrapper_type", attr=VerifyKeyWrapper
)


class VerifyAllWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> VerifyAllWrapper_PB:
        return VerifyAllWrapper_PB(all=Empty_PB())

    @staticmethod
    def _data_proto2object(proto: VerifyAllWrapper_PB) -> VerifyAll:  # type: ignore
        return VerifyAll()

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return VerifyAllWrapper_PB

    @staticmethod
    def get_wrapped_type() -> Type:
        return VerifyAll

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=VerifyAll, name="serializable_wrapper_type", attr=VerifyAllWrapper
)
