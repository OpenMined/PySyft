# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...proto.core.common.common_object_pb2 import ObjectWithID as ObjectWithID_PB
from ...util import validate_type
from .serde.deserialize import _deserialize
from .serde.serializable import serializable
from .serde.serialize import _serialize as serialize
from .uid import UID


@serializable()
class ObjectWithID:
    """This object is the superclass for nearly all Syft objects. Subclassing
    from this object will cause an object to be initialized with a unique id
    using the process specified in the UID class.

    .. note::
        At the time of writing, the only class in Syft which doesn't have an ID
        of some kind is the Client class because it's job is to point to another
        object (which has an ID).

    .. note::
        Be aware of performance choices in this class because it is used so
        heavily across the entire codebase. Assume every method is going to
        be called thousands of times during the working day of an average
        data scientist using syft (and millions of times in the context of a
        machine learning job).

    """

    def __init__(self, id: Optional[UID] = None):
        """This initializer only exists to set the id attribute, which is the
        primary purpose of this class. It also sets the 'as_wrapper' flag
        for the 'Serializable' superclass.

        Args:
            id: an override which can be used to set an ID for this object

        """

        if id is None:
            id = UID()

        self._id: UID = id

        # while this class is never used as a simple wrapper,
        # it's possible that sub-classes of this class will be.
        super().__init__()

    @property
    def id(self) -> UID:
        """We reveal ObjectWithID.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        Returns:
            returns the unique id of the object
        """
        return self._id

    def __eq__(self, other: Any) -> bool:
        """Checks to see if two ObjectWithIDs are actually the same object.

        This checks to see whether this ObjectWithIDs is equal to another by
        comparing whether they have the same .id objects. These objects
        come with their own __eq__ function which we assume to be correct.

        Args:
            other: this is the other ObjectWithIDs to be compared with

        Returns:
            True/False based on whether the objects are the same
        """

        try:
            return self.id == other.id
        except Exception:
            return False

    def __repr__(self) -> str:
        """
        Return a human-readable representation of the ObjectWithID with brackets
        so that it can be easily spotted when nested inside of the human-
        readable representations of other objects.

        Returns:
            a human-readable version of the ObjectWithID

        """

        no_dash = str(self.id.value).replace("-", "")
        return f"<{type(self).__name__}: {no_dash}>"

    def repr_short(self) -> str:
        """
        Return a SHORT human-readable version of the ID which
        makes it print nicer when embedded (often alongside other
        UID objects) within other object __repr__ methods.

        Returns:
            a SHORT human-readable version of SpecificLocation
        """

        return f"<{type(self).__name__}:{self.id.repr_short()}>"

    def _object2proto(self) -> ObjectWithID_PB:
        """
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        Returns:
            a protobuf object that is the serialization of self.

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return ObjectWithID_PB(id=serialize(self.id))

    @staticmethod
    def _proto2object(proto: ObjectWithID_PB) -> "ObjectWithID":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        Args:
            proto: a protobuf object that we wish to convert to instance of this class

        Returns:
            an instance of ObjectWithID

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        _id = validate_type(_object=_deserialize(proto.id), _type=UID, optional=True)
        return ObjectWithID(id=_id)

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

        Returns:
            the type of protobuf object which corresponds to this class.

        """

        return ObjectWithID_PB
