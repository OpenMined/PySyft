"""In this test suite, we evaluate the Address class. For more info
on the Address class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways Address can/can't be initialized
    - CLASS METHODS: tests for the use of Address's class methods
    - SERDE: test for serialization and deserialization of Address.

"""


# stdlib
from itertools import combinations
import uuid

# third party
import pytest

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.location.specific import SpecificLocation

ARGUMENTS = ["vm", "device", "domain", "network"]

# --------------------- INITIALIZATION ---------------------


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
    Helper method to generate pre-ordered arguments for initializing an Address instance.
    There are at least 3 arguments, all taken from 'vm', 'device', 'domain', 'network'.
    """
    # the order matches the _gen_icons below
    all_combos = []
    for combination in combinations(ARGUMENTS, 3):
        all_combos.append(list(combination))
    all_combos.append(ARGUMENTS)
    return [{key: SpecificLocation(id=UID()) for key in combo} for combo in all_combos]


def _gen_icons() -> list:
    """Helper method to return an pre-ordered list of icons."""
    return [
        "💠 [🍰📱🏰]",
        "💠 [🍰📱🔗]",
        "💠 [🍰🏰🔗]",
        "💠 [📱🏰🔗]",
        "💠 [🍰📱🏰🔗]",
    ]


def _gen_address_kwargs_and_expected_values() -> list:
    """
    Helper method to generate kwargs for initializing Address as well as
    the expected values thereof.
    """
    address_kwargs = _gen_address_kwargs()
    expected_value_dict = [
        {key: kwargs.get(key, None) for key in ARGUMENTS} for kwargs in address_kwargs
    ]
    return list(zip(address_kwargs, expected_value_dict))


@pytest.mark.parametrize(
    "address_kwargs, expected_values", _gen_address_kwargs_and_expected_values()
)
def test_init_with_specific_id(address_kwargs: dict, expected_values: dict) -> None:
    """Test that Address will use the SpecificLocation you pass into the constructor"""
    address = Address(**address_kwargs)

    assert address.network is expected_values["network"]
    assert address.domain is expected_values["domain"]
    assert address.device is expected_values["device"]
    assert address.vm is expected_values["vm"]


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


def test_target_emoji_method() -> None:
    """Unit test for Address.target_emoji method"""
    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    address = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )
    assert address.target_emoji() == "@<UID:🙍🛖>"


# --------------------- PROPERTY METHODS ---------------------


@pytest.mark.parametrize(
    "address_kwargs, expected_icon",
    list(
        zip(
            _gen_address_kwargs(),
            _gen_icons(),
        )
    ),
)
def test_icon_property_method(address_kwargs: dict, expected_icon: str) -> None:
    """Unit tests for Address.icon property method"""
    address = Address(**address_kwargs)
    assert address.icon == expected_icon


@pytest.mark.parametrize(
    "address_kwargs, expected_icon",
    list(
        zip(
            _gen_address_kwargs(),
            _gen_icons(),
        )
    ),
)
def test_pprint_property_method(address_kwargs: dict, expected_icon: str) -> None:
    """Unit tests for Address.pprint property method"""
    named_address = Address(name="Sneaky Nahua", **address_kwargs)
    assert named_address.pprint == expected_icon + " Sneaky Nahua (Address)"

    unnamed_address = Address(**address_kwargs)
    assert expected_icon in unnamed_address.pprint
    assert "(Address)" in unnamed_address.pprint


def test_address_property_method() -> None:
    """Unit tests for Address.address property method"""
    address = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    returned_address = address.address
    assert isinstance(returned_address, Address)
    assert returned_address.network == address.network
    assert returned_address.domain == address.domain
    assert returned_address.device == address.device
    assert returned_address.vm == address.vm


def test_network_getter_and_setter() -> None:
    """Unit tests for Address.network getter and setter"""
    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    # Test getter
    network = SpecificLocation(id=an_id)
    address = Address(
        network=network,
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )
    assert address.network == network

    # Test setter
    new_network = SpecificLocation(id=an_id)
    address.network = new_network
    assert address.network == new_network


def test_network_id_property_method() -> None:
    """Unit test for Address.network_id method"""
    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    # Test getter
    address_with_network = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )
    address_without_network = Address(
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    assert address_with_network.network_id == an_id
    assert address_without_network.network_id is None


def test_domain_and_domain_id_property_methods() -> None:
    """Unit test for Address.domain and Address.domain_id methods"""
    # Test getter
    domain = SpecificLocation(id=UID())
    address_with_domain = Address(
        network=SpecificLocation(id=UID()),
        domain=domain,
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )
    # Test domain getter
    assert address_with_domain.domain == domain

    # Test domain setter
    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    new_domain = SpecificLocation(id=an_id)
    address_with_domain.domain = new_domain
    assert address_with_domain.domain == new_domain

    # Test domain_id getter
    address_without_domain = Address(
        network=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )
    assert address_with_domain.domain_id == an_id
    assert address_without_domain.domain_id is None


def test_device_and_device_id_property_methods() -> None:
    """Unit test for Address.device and Address.device_id methods"""
    # Test getter
    device = SpecificLocation(id=UID())
    address_with_device = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=device,
        vm=SpecificLocation(id=UID()),
    )
    # Test device getter
    assert address_with_device.device == device

    # Test device setter
    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    new_device = SpecificLocation(id=an_id)
    address_with_device.device = new_device
    assert address_with_device.device == new_device

    # Test domain_id getter
    address_without_device = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )
    assert address_with_device.device_id == an_id
    assert address_without_device.device_id is None


def test_vm_and_vm_id_property_methods() -> None:
    """Unit test for Address.vm and Address.vm_id methods"""
    # Test getter
    vm = SpecificLocation(id=UID())
    address_with_vm = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=vm,
    )
    # Test device getter
    assert address_with_vm.vm == vm

    # Test device setter
    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    new_vm = SpecificLocation(id=an_id)
    address_with_vm.vm = new_vm
    assert address_with_vm.vm == new_vm

    # Test domain_id getter
    address_without_vm = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
    )
    assert address_with_vm.vm_id == an_id
    assert address_without_vm.vm_id is None


def test_target_id_property_method_with_a_return() -> None:
    """Unit test for Address.target_id method"""
    network = SpecificLocation(id=UID())
    domain = SpecificLocation(id=UID())
    device = SpecificLocation(id=UID())
    vm = SpecificLocation(id=UID())
    address = Address(
        network=network,
        domain=domain,
        device=device,
        vm=vm,
    )
    assert address.target_id == vm
    address.vm = None
    assert address.target_id == device
    address.device = None
    assert address.target_id == domain
    address.domain = None
    assert address.target_id == network


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

    blob = sy.serialize(obj, to_proto=True)

    assert sy.serialize(obj) == blob
    assert obj == sy.deserialize(blob=blob)


def test_partial_serialization_and_deserialization() -> None:
    """Test that addresses with only some attributes serialize and deserialize correctly/"""

    an_id = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    obj = Address(  # network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    assert obj == sy.deserialize(blob=sy.serialize(obj))

    obj = Address(
        network=SpecificLocation(id=an_id),
        # domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    blob = sy.serialize(obj, to_proto=True)
    assert obj == sy.deserialize(blob=blob)

    obj = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        # device=SpecificLocation(id=an_id),
        vm=SpecificLocation(id=an_id),
    )

    blob = sy.serialize(obj, to_proto=True)
    assert obj == sy.deserialize(blob=blob)

    obj = Address(
        network=SpecificLocation(id=an_id),
        domain=SpecificLocation(id=an_id),
        device=SpecificLocation(id=an_id),
        # vm=SpecificLocation(id=an_id)
    )

    blob = sy.serialize(obj, to_proto=True)
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
        network=sy.serialize(loc),
        domain=sy.serialize(loc),
        device=sy.serialize(loc),
        vm=sy.serialize(loc),
    )

    assert sy.serialize(obj, to_proto=True) == blob
    assert sy.serialize(obj, to_proto=True) == blob
    assert sy.serialize(obj, to_proto=True) == blob


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
        network=sy.serialize(loc),
        domain=sy.serialize(loc),
        device=sy.serialize(loc),
        vm=sy.serialize(loc),
    )

    obj2 = sy.deserialize(blob=blob, from_proto=True)
    assert obj == obj2
