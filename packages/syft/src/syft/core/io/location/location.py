# stdlib
from typing import Optional

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ....logger import traceback_and_raise
from ....util import random_name
from ...common.uid import UID


class Location:
    """This represents the location of a node, including
    location-relevant metadata (such as how long it takes
    for us to communicate with this location, etc.)"""

    def __init__(self, name: Optional[str] = None) -> None:
        if name is None:
            name = random_name()
        self.name = name
        super().__init__()

    @property
    def id(self) -> UID:
        traceback_and_raise(NotImplementedError)

    def repr_short(self) -> str:
        """Returns a SHORT human-readable version of the ID

        Return a SHORT human-readable version of the ID which
        makes it print nicer when embedded (often alongside other
        UID objects) within other object __repr__ methods."""

        return self.__repr__()

    @staticmethod
    def _proto2object(proto: Message) -> "Location":
        """This method converts a protobuf object into a subclass of Serializable

        This method must be implemented for all classes which subclassSerializable - namely
        all classes which can be serialized within the Syft ecosystem. It should convert the
        corresponding protobuf message for the subclass into an instance of the class. This
        allows all the logic which goes from protobuf message to other formats (JSON, binary, etc.)
        to be generic and simply inherited from the rest of this class.

        :param proto: the protobuf object to be converted into an instance of type(self)
        :param type: Message
        :return: an instance of type(self)
        :rtype: Serializable

        """
        traceback_and_raise(NotImplementedError)

    def _object2proto(self) -> Message:
        """This methods converts self into a protobuf object

        This method must be implemented by all subclasses so that generic high-level functions
        implemented here (such as serialize(, to_bytes=True), etc) know how to convert the object into
        a protobuf object before further converting it into the requested format.

        :return: a protobuf message
        :rtype: Message
        """

        traceback_and_raise(NotImplementedError)

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

        traceback_and_raise(NotImplementedError)
