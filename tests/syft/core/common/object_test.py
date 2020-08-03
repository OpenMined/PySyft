"""In this test suite, we evaluate the ObjectWithID class. For more info
on the ObjectWithID class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways ObjectWithID can/can't be initialized
    - CLASS METHODS: tests for the use of ObjectWithID's class methods
    - SERDE: test for serialization and deserialization of ObjectWithID.
    - CHILDREN: test that subclasses of ObjectWithID fulfill standards

"""

# external imports
import uuid
import pytest

# syft imports
import syft as sy
from syft.core.common import ObjectWithID
from syft.util import get_subclasses
from syft.core.common import UID

################## INITIALIZATION ######################


def test_basic_init():
    """Test that creating ObjectWithID() does in fact create
    an object with an id."""

    obj = ObjectWithID()
    assert isinstance(obj.id, UID)


def test_immutability_of_id():
    """We shouldn't allow people to modify the id of an
    ObjectWithID because this can create all sorts of errors.

    Put the other way around - blocking people from modifying
    the ID of an object means we can build a codebase which more
    firmly relies on the id being truthful. It also will avoid
    people initialising objects in weird ways (setting ids later).
    """
    obj = ObjectWithID()

    with pytest.raises(AttributeError) as e:

        # TODO: filter on this error to only include errors
        #  with string "Can't set attribute"

        obj.id = ""


################## CLASS METHODS #######################


def test_compare():
    """While uses of this feature should be quite rare, we
    should be able to check whether two objects are the same
    based on their IDs being the same by default. Note that
    subclasses will undoubtedly modify this behavior with other
    __eq__ methods."""

    obj = ObjectWithID()
    obj2 = ObjectWithID()

    assert obj != obj2

    obj._id = obj2.id

    assert obj == obj2


def test_to_string():
    """Tests that UID generates an intuitive string."""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    assert str(obj) == "<ObjectWithID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"
    assert obj.__repr__() == "<ObjectWithID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"


###################### SERDE ##########################
###################### CHILDREN ##########################


def test_subclasses_have_names():
    """Ensure that all known subclassses of ObjectWithID have
    a __name__ parameter. I'm not sure why but occasionally
    I came across objects without them"""

    subclasses = get_subclasses(obj_type=ObjectWithID)

    for sc in subclasses:
        assert hasattr(sc, "__name__")


def test_subclasses_of_obj_with_id_have_their_own_protobuf_types_with_correct_names():
    """Ensure that all known subclassses of ObjectWithID have
    a custom protobuf_type parameter. This could be easy to
    accidentally forget to add.

    The reason this is useful is that since ObjectWithID has a type
    all subclasses will inherit it even if they have more things
    to serialize. This could results in annoying dev experiences
    if people don't know to add this flag. So, we'll create a test
    just to check!

    Specifically, this test ENFORCES that all protobuf type names must
    have the SAME NAME (the same .__name__) as the object they are
    supposed to serialize.
    """

    # TODO: write protobufs for these objects and remove them from this test.
    known_exceptions = {
        "Location",
        "LocationGroup",
        "SubscriptionBackedLocationGroup",
        "RegistryBackedLocationGroup",
        "AbstractNode",
        "Node",
        "VirtualMachine",
        "Device",
        "Domain",
        "Network",
        "RouteSchema",
        "Route",
        "BroadcastRoute",
        "SoloRoute",
    }

    subclasses = get_subclasses(obj_type=ObjectWithID)

    for sc in subclasses:
        if sc.__name__ not in known_exceptions:

            # Assert that each protobuf type's name is the same
            # as the object it is intended to serialize
            assert sc.protobuf_type.__name__ == sc.__name__
