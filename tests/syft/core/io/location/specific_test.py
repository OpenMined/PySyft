"""In this test suite, we evaluate the SpecificLocation class. For more info
on the SpecificLocation class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways UID can/can't be initialized
    - CLASS METHODS: tests for the use of UID's class methods
    - SERDE: test for serialization and deserialization of UID.

"""

# external imports
import uuid

# syft imports
from syft.core.io.location.specific import SpecificLocation
from syft.core.common.uid import UID

# --------------------- INITIALIZATION ---------------------


def test_specific_location_init_without_arguments():
    """Test that SpecificLocation will self-create an ID object if none is given"""

    # init works without arguments
    loc = SpecificLocation()

    assert isinstance(loc.id, UID)


def test_specific_location_init_with_specific_id():
    """Test that SpecificLocation will use the ID you pass into the constructor"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    loc = SpecificLocation(id=uid)

    assert loc.id == uid


# --------------------- CLASS METHODS ---------------------
# --------------------- SERDE ---------------------
