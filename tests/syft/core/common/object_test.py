"""In this test suite, we evaluate the ObjectWithID class. For more info
on the ObjectWithID class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways ObjectWithID can/can't be initialized
    - CLASS METHODS: tests for the use of ObjectWithID's class methods
    - SERDE: test for serialization and deserialization of ObjectWithID.

"""

# external imports
import pytest

# syft imports
import syft as sy
from syft.core.common import UID
from syft.core.common import ObjectWithID

################## INITIALIZATION ######################


def test_basic_init():
    """Test that creating ObjectWithID() does in fact create
    an object with an id."""

    obj = ObjectWithID()
    assert isinstance(obj.id, UID)

#
# def test_immutability_of_id():
#     """We shouldn't allow people to modify the id of an
#     ObjectWithID because this can create all sorts of errors.
#
#     Put the other way around - blocking people from modifying
#     the ID of an object means we can build a codebase which more
#     firmly relies on the id being truthful. It also will avoid
#     people initialising objects in weird ways (setting ids later).
#     """
#
#
# def test_compare():
#
#     obj = ObjectWithID()
#     obj2 = ObjectWithID()
#
#     assert obj != obj2
#
#     obj.id = obj2.id
#
#     assert obj == obj2

################## CLASS METHODS #######################
###################### SERDE ##########################