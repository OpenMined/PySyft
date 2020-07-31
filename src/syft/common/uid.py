import uuid
from typing import final
from ..proto import ProtoUID
from ..decorators.syft_decorator_impl import syft_decorator

@final
class UID(object):
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

    @syft_decorator(typechecking=True)
    def __init__(self, value:bool = None):
        """This initializes the object. Normal use for this object is
        to initialize the constructor with value==None because you
        want to initialize with a novel ID. The only major exception
        is deserialization, wherein a UID object is created with a
        specific id value."""

        # if value is not set - create a novel and unique ID.
        if value is None:

            # for more info on how this UUID is generated:
            # https://docs.python.org/2/library/uuid.html
            value = uuid.uuid4()

        # save the ID's value. Note that this saves the uuid value
        # itself instead of saving the
        self.value = value

    def __hash__(self):
        """A very common use of UID objects is as a key in a dictionary
        or database. The object must be able to be hashed in order to
        be used in this way. We take the 128-bit int representation of the
        value. Note that this probably gets further hashed into a shorter
        representation for most python data-structures.

        Note that we assume that any collisions will be very rare and
        detected by the ObjectStore class in Syft."""

        return self.value.int

    def __eq__(self, other):
        if isinstance(other, UID):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"<UID:{self.value}>"

    def serialize(self):
        return ProtoUID(value=self.value.bytes)

@staticmethod
def deserialize(proto_uid: ProtoUID) -> UID:
    return UID(value=uuid.UUID(bytes=proto_uid.value))

# We first create the UID class and then add this static
# method so that we can add the UID class as a return
# type hint directly.
UID.deserialize = deserialize