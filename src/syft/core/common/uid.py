# external lib imports
import uuid

# external class/method imports
from typing import final

# syft imports
from ...decorators.syft_decorator_impl import syft_decorator
from .serializable import Serializable
from ...proto import ProtoUID

# resources
uuid_type = type(uuid.uuid4())


@final
class AbstractUID(Serializable):
    """This exists to allow us to typecheck on the UID object
    """


@final
class UID(AbstractUID):
    """This object creates a unique ID for every object in the Syft
    ecosystem. This ID is guaranteed to be unique for the node on
    which it is initialized and is very likely to be unique across
    the whole ecosystem (because it is long and randomly generated).

    Nearly all objects within Syft subclass from this object because
    nearly all objects need to have a unique ID. The only major
    exception a the time of writing is the Client object because it
    just points to another object which itself has an id.

    There is no other way in Syft to create an ID for any object.

    """

    proto_type = ProtoUID

    @syft_decorator(typechecking=True)
    def __init__(self, value: uuid_type = None):
        """This initializes the object. Normal use for this object is
        to initialize the constructor with value==None because you
        want to initialize with a novel ID. The only major exception
        is deserialization, wherein a UID object is created with a
        specific id value.

        :param value: if you want to initialize an object with a specific UID, pass it in here. This is normally only used during deserialization.
        :type value: uuid.uuid4(), optional
        :return: returns the initialized object
        :rtype: UID

        .. code-block:: python

            from syft.core.common.uid import UID
            my_id = UID()
            print(my_id.value)

        .. code-block:: bash

            >>> 8d744978-327b-4126-a644-cb90bcadd35e
        """
        # checks to make sure you've set a proto_type
        super().__init__()

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
        """A very common use of UID objects is as a key in a dictionary
        or database. The object must be able to be hashed in order to
        be used in this way. We take the 128-bit int representation of the
        value.

        :param value (uuid): if you want to initialize an object with a specific UID, pass it in here. This is normally only used during deserialization.
        :param arg2: description
        :type arg1: type description
        :type arg1: type description
        :return: return description
        :rtype: the return type description

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
        """This checks to see whether this UID is equal to another UID by
        comparing whether they have the same .value objects. These objects
        come with their own __eq__ function which we assume to be correct.
    """

        if isinstance(other, UID):
            return self.value == other.value

    def __repr__(self):
        return f"<UID:{self.value}>"

    def object2proto(self):
        self_type = type(self)
        obj_type = self_type.__module__ + "." + self_type.__name__
        return ProtoUID(obj_type=obj_type, value=self.value.bytes)

    @staticmethod
    def proto2object(proto: ProtoUID) -> AbstractUID:
        return UID(value=uuid.UUID(bytes=proto.value))
