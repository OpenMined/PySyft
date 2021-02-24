# stdlib
from typing import Any
from typing import Optional
import uuid
from uuid import UUID as uuid_type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...logger import critical
from ...logger import traceback_and_raise
from ...proto.core.common.common_object_pb2 import UID as UID_PB
from ..common.serde.serializable import Serializable
from ..common.serde.serializable import bind_protobuf


@bind_protobuf
class UID(Serializable):
    """A unique ID for every Syft object.

    This object creates a unique ID for every object in the Syft
    ecosystem. This ID is guaranteed to be unique for the node on
    which it is initialized and is very likely to be unique across
    the whole ecosystem (because it is long and randomly generated).

    Nearly all objects within Syft subclass from this object because
    nearly all objects need to have a unique ID. The only major
    exception a the time of writing is the Client object because it
    just points to another object which itself has an id.

    There is no other way in Syft to create an ID for any object.

    """

    value: uuid_type

    def __init__(self, value: Optional[uuid_type] = None):
        """Initializes the internal id using the uuid package.

        This initializes the object. Normal use for this object is
        to initialize the constructor with value==None because you
        want to initialize with a novel ID. The only major exception
        is deserialization, wherein a UID object is created with a
        specific id value.

        :param value: if you want to initialize an object with a specific UID, pass it
                      in here. This is normally only used during deserialization.
        :type value: uuid.uuid4, optional
        :return: returns the initialized object
        :rtype: UID

        .. code-block:: python

            from syft.core.common.uid import UID
            my_id = UID()
        """
        # checks to make sure you've set a proto_type
        super().__init__()

        # if value is not set - create a novel and unique ID.
        if value is None:

            # for more info on how this UUID is generated:
            # https://docs.python.org/3/library/uuid.html
            value = uuid.uuid4()

        # save the ID's value. Note that this saves the uuid value
        # itself instead of saving the
        self.value = value

    @staticmethod
    def from_string(value: str) -> "UID":
        try:
            return UID(value=uuid.UUID(value))
        except Exception as e:
            critical(f"Unable to convert {value} to UUID. {e}")
            traceback_and_raise(e)

    def __hash__(self) -> int:
        """Hashes the UID for use in dictionaries and sets

        A very common use of UID objects is as a key in a dictionary
        or database. The object must be able to be hashed in order to
        be used in this way. We take the 128-bit int representation of the
        value.

        :return: returns a hash of the object
        :rtype: int

        .. note::
            Note that this probably gets further hashed into a shorter
            representation for most python data-structures.

        .. note::
            Note that we assume that any collisions will be very rare and
            detected by the ObjectStore class in Syft.
        """

        return self.value.int

    def __eq__(self, other: Any) -> bool:
        """Checks to see if two UIDs are the same using the internal object

        This checks to see whether this UID is equal to another UID by
        comparing whether they have the same .value objects. These objects
        come with their own __eq__ function which we assume to be correct.

        :param other: this is the other ID to be compared with
        :type other: Any (note this must be Any or __eq__ fails on other types)
        :return: returns True/False based on whether the objects are the same
        :rtype: bool
        """

        try:
            return self.value == other.value
        except Exception:
            return False

    def __repr__(self) -> str:
        """Returns a human-readable version of the ID

        Return a human-readable representation of the UID with brackets
        so that it can be easily spotted when nested inside of the human-
        readable representations of other objects."""

        no_dash = str(self.value).replace("-", "")
        return f"<{type(self).__name__}: {no_dash}>"

    def char_emoji(self, hex_chars: str) -> str:
        base = ord("\U0001F642")
        hex_base = ord("0")
        code = 0
        for char in hex_chars:
            offset = ord(char)
            code += offset - hex_base
        return chr(base + code)

    def string_emoji(self, string: str, length: int, chunk: int) -> str:
        output = []
        part = string[-length:]
        while len(part) > 0:
            part, end = part[:-chunk], part[-chunk:]
            output.append(self.char_emoji(hex_chars=end))
        return "".join(output)

    def emoji(self) -> str:
        return f"<UID:{self.string_emoji(string=str(self.value), length=8, chunk=4)}>"

    def repr_short(self) -> str:
        """Returns a SHORT human-readable version of the ID

        Return a SHORT human-readable version of the ID which
        makes it print nicer when embedded (often alongside other
        UID objects) within other object __repr__ methods."""

        return f"..{str(self.value)[-5:]}"

    def _object2proto(self) -> UID_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ProtoUID

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return UID_PB(value=self.value.bytes)

    @staticmethod
    def _proto2object(proto: UID_PB) -> "UID":
        """Creates a UID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of UID
        :rtype: UID

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return UID(value=uuid.UUID(bytes=proto.value))

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
        return UID_PB
