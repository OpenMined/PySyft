"""In this test suite, we evaluate the Address class. For more info
on the Address class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways Address can/can't be initialized
    - CLASS METHODS: tests for the use of Address's class methods
    - SERDE: test for serialization and deserialization of Address.

"""


# stdlib
import uuid
from itertools import combinations

# third party
import pytest

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.location.specific import SpecificLocation

# --------------------- INITIALIZATION ---------------------
ARGUMENTS = ('vm', 'device', 'domain', 'network')

def test_init_without_arguments() -> None:
    """Test that Address have all attributes as None if none are given"""

    # init works without arguments
    addr = Address()

    assert addr.network is None
    assert addr.domain is None
    assert addr.device is None
    assert addr.vm is None

    with pytest.raises(Exception):
        assert addr.target_id is None


def _gen_address_kwargs() -> list:
    """
    Helper method to generate 3-4 arguments for initializing an Address instance.
    Arguments are taken from 'vm', 'device', 'domain', 'network'.
    """
    all_combos = list(combinations(ARGUMENTS, 3)) + [ARGUMENTS]
    return [
        {key: SpecificLocation(id=UID()) for key in combo}
        for combo in all_combos
    ]


def _gen_address_kwargs_and_expected_values() -> list:
    """
    Helper method to generate kwargs for initializing Address as well as
    the expected values thereof.
    """
    address_kwargs = _gen_address_kwargs()
    expected_value_dict = [
        {
            key: kwargs.get(key, None)
            for key in ARGUMENTS
        }
        for kwargs in address_kwargs
    ]
    return list(zip(address_kwargs, expected_value_dict))


@pytest.mark.parametrize(
    'address_kwargs, expected_values', _gen_address_kwargs_and_expected_values()
)
def test_init_with_specific_id(address_kwargs: dict, expected_values: dict) -> None:
    """Test that Address will use the SpecificLocation you pass into the constructor"""
    addr = Address(**address_kwargs)

    assert addr.network is expected_values['network']
    assert addr.domain is expected_values['domain']
    assert addr.device is expected_values['device']
    assert addr.vm is expected_values['vm']


# --------------------- CLASS METHODS ---------------------


def test_compare() -> None:
    """Tests whether two address objects are the same. This functionality
    is likely to get used a lot when nodes are determining whether a message
    is for them or not."""

    x = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    y = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    z = Address(network=x.network, domain=x.domain, device=x.device, vm=x.vm)

    assert x != y
    assert x == z
    assert y != z


def test_to_string() -> None:
    """Tests that SpecificLocation generates an intuitive string."""

    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    obj = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    str_out = (
        "<Address - Network:<SpecificLocation:..a1514>, Domain:<Specific"
        "Location:..a1514>  Device:<SpecificLocation:..a1514>, VM:<Specific"
        "Location:..a1514>"
    )

    assert str(obj) == str_out
    assert obj.__repr__() == str_out


@pytest.mark.parametrize(
    'address_kwargs, expected_icon', list(zip(
        _gen_address_kwargs(),
        [
            '💠 [🍰📱🏰]',
            '💠 [🍰📱🔗]',
            '💠 [🍰🏰🔗]',
            '💠 [📱🏰🔗]',
            '💠 [🍰📱🏰🔗]',
        ]
    ))
)
def test_icon(address_kwargs, expected_icon) -> None:
    """Test that Address.icon property method returns the correct emojis."""
    addr = Address(**address_kwargs)
    assert addr.icon == expected_icon


# --------------------- SERDE ---------------------


def test_default_serialization_and_deserialization() -> None:
    """Tests that default Address serde works as expected - to Protobuf"""

    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    obj = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    blob = obj.to_proto()

    assert obj.serialize() == blob
    assert obj == sy.deserialize(blob=blob)


def test_partial_serialization_and_deserialization() -> None:
    """Test that addresses with only some attributes serialize and deserialize correctly/"""

    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    obj = Address(  # network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    assert obj == sy.deserialize(blob=obj.serialize())

    obj = Address(
        network=SpecificLocation(id=an_id),
        # domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    blob = obj.to_proto()
    assert obj == sy.deserialize(blob=blob)

    obj = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        # device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    blob = obj.to_proto()
    assert obj == sy.deserialize(blob=blob)

    obj = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        # vm=SpecificLocation(id=an_id)
    )

    blob = obj.to_proto()
    assert obj == sy.deserialize(blob=blob)


def test_proto_serialization() -> None:
    """Tests that default Address serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    loc = SpecificLocation(id=uid, name="Test Location")
    obj = Address(
        name="Test Address",
        network=loc,
        domain=loc,
        device=loc,
        vm=loc,
    )

    blob = Address.get_protobuf_schema()(
        name="Test Address",
        has_network=True,
        has_domain=True,
        has_device=True,
        has_vm=True,
        network=loc.serialize(),
        domain=loc.serialize(),
        device=loc.serialize(),
        vm=loc.serialize(),
    )

    assert obj.proto() == blob
    assert obj.to_proto() == blob
    assert obj.serialize(to_proto=True) == blob


def test_proto_deserialization() -> None:
    """Tests that default Address deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    loc = SpecificLocation(id=uid, name="Test Location")

    obj = Address(
        network=loc,
        domain=loc,
        device=loc,
        vm=loc,
    )

    blob = Address.get_protobuf_schema()(
        has_network=True,
        has_domain=True,
        has_device=True,
        has_vm=True,
        network=loc.serialize(),
        domain=loc.serialize(),
        device=loc.serialize(),
        vm=loc.serialize(),
    )

    obj2 = sy.deserialize(blob=blob, from_proto=True)
    assert obj == obj2
