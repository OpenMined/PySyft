"""In this test suite, we evaluate the SpecificLocation class. For more info
on the SpecificLocation class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways UID can/can't be initialized
    - CLASS METHODS: tests for the use of UID's class methods
    - SERDE: test for serialization and deserialization of UID.

"""

# syft imports
from syft.core.io.location.specific import SpecificLocation
from syft.core.common.uid import UID

# --------------------- INITIALIZATION ---------------------


def test_specific_location_init_without_arguments():

    # init works without arguments
    loc = SpecificLocation()

    assert isinstance(loc.id, UID)


# --------------------- CLASS METHODS ---------------------
# --------------------- SERDE ---------------------
