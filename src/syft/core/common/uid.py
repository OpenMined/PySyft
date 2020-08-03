# external lib imports
import uuid

# external class/method imports
from syft.core.common.serializable import Serializable
from typing import final

# syft imports
from ...proto.core.common.common_object_pb2 import UID as UID_PB
from ...decorators import syft_decorator

# resources
uuid_type = type(uuid.uuid4())


@final
class AbstractUID(Serializable):
    """This exists to allow us to typecheck on the UID object
    because we need a type which has already been initialized in
    order to add it as a type hint on the UID object.
    """


@final
class UID(AbstractUID):
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

    protobuf_type = UID_PB
    is_wrapper = True
    wrapping_class = uuid_type

    @syft_decorator(typechecking=True)
    def __init__(self, value: uuid_type = None, as_wrapper:bool = False):
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
        super().__init__(as_wrapper=as_wrapper)

        # if value is not set - create a novel and unique ID.
        if value is None:

            # for more info on how this UUID is generated:
            # https://docs.python.org/2/library/uuid.html
            value = uuid.uuid4()

        # save the ID's value. Note that this saves the uuid value
        # itself instead of saving the
        self.value = value

    @syft_decorator(typechecking=True)
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

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: AbstractUID) -> bool:
        """Checks to see if two UIDs are the same using the internal object

        This checks to see whether this UID is equal to another UID by
        comparing whether they have the same .value objects. These objects
        come with their own __eq__ function which we assume to be correct.

        :param other: this is the other ID to be compared with
        :type other: AbstractUID
        :return: returns True/False based on whether the objcts are the same
        :rtype: bool
        """

        if isinstance(other, UID):
            return self.value == other.value

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        """Returns a human-readable version of the ID

        Return a human-readable representation of the UID with brackets
        so that it can be easily spotted when nested inside of the human-
        readable representations of other objects."""

        return f"<UID:{self.value}>"

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> UID_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ProtoUID

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        self_type = type(self)
        obj_type = self_type.__module__ + "." + self_type.__name__
        return UID_PB(
            obj_type=obj_type, value=self.value.bytes, as_wrapper=self.as_wrapper
        )

    @staticmethod
    def _proto2object(proto: UID_PB) -> AbstractUID:
        """Creates a UID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of UID
        :rtype: UID

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        value = uuid.UUID(bytes=proto.value)
        if proto.as_wrapper:
            return value
        return UID(value=value)


# This flag is what allows the serializer to find this class
# when it encounters an object of uuid_type.
uuid_type.serializable_wrapper_type = UID
