# stdlib
from typing import Any
from typing import Optional

# relative
from .serde.serializable import serializable
from .uid import UID


@serializable(recursive_serde=True)
class ObjectWithID:
    __attr_allowlist__ = ("_id",)

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

    @id.setter
    def id(self, new_id: UID) -> None:
        self._id = new_id

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
