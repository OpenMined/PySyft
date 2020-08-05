from __future__ import annotations

# external class/method imports (sorted by length)
from ...proto.core.common.common_object_pb2 import ObjectWithID as ObjectWithID_PB

# syft imports (sorted by length)
from ...decorators.syft_decorator_impl import syft_decorator
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.serde.serializable import Serializable
from .uid import UID


class ObjectWithID(Serializable):
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

    @syft_decorator(typechecking=True)
    def __init__(self, id: UID = None, as_wrapper: bool = False):
        """This initializer only exists to set the id attribute, which is the
        primary purpose of this class. It also sets the 'as_wrapper' flag
        for the 'Serializable' superclass.

        :param id: an override which can be used to set an ID for this object
            manually. This is probably only used for deserialization.
        :type id: UID
        :param as_wrapper: this flag determines whether the subclass can also
            be used as a wrapper class around a non-syft object. For details on
            why, see :py:mod:`syft.core.common.serializable.Serializable`.

        """

        # while this class is never used as a simple wrapper,
        # it's possible that sub-classes of this class will be.
        super().__init__()

        if id is None:
            id = UID()

        self._id = id

    @property
    def id(self) -> UID:
        """We reveal ObjectWithID.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: "ObjectWithID") -> bool:
        """Checks to see if two ObjectWithIDs are actually the same object.

        This checks to see whether this ObjectWithIDs is equal to another by
        comparing whether they have the same .id objects. These objects
        come with their own __eq__ function which we assume to be correct.

        :param other: this is the other ObjectWithIDs to be compared with
        :type other: ObjectWithIDs
        :return: returns True/False based on whether the objcts are the same
        :rtype: bool
        """

        return self.id == other.id

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        """Returns a human-readable version of the ObjectWithID

        Return a human-readable representation of the ObjectWithID with brackets
        so that it can be easily spotted when nested inside of the human-
        readable representations of other objects."""

        return f"<{ObjectWithID.__name__}:{self.id.value}>"

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> ObjectWithID_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectWithID_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return ObjectWithID_PB(id=self.id.serialize())

    @staticmethod
    def _proto2object(proto: ObjectWithID_PB) -> "ObjectWithID":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ObjectWithID
        :rtype: ObjectWithID

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return ObjectWithID(id=_deserialize(blob=proto.id))

    @staticmethod
    def get_protobuf_schema() -> type:
        return ObjectWithID_PB