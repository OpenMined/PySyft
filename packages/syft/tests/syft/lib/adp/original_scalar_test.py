# syft absolute
from syft import deserialize
from syft import serialize
from syft.lib.adp.entity import Entity
from syft.lib.adp.scalar import OriginScalar


def test_scalar() -> None:
    bob = OriginScalar(value=1, min_val=-2, max_val=2, entity=Entity(unique_name="Bob"))
    alice = OriginScalar(
        value=1, min_val=-1, max_val=1, entity=Entity(unique_name="Alice")
    )
    bob + alice


def test_required_serde() -> None:
    bob = OriginScalar(value=1, min_val=-2, max_val=2, entity=Entity(unique_name="Bob"))

    serialized = serialize(bob, to_bytes=True)
    deserialized = deserialize(serialized, from_bytes=True)

    assert bob.id == deserialized.id
    assert bob.name == deserialized.name
    assert bob.value == deserialized.value
    assert bob.min_val == deserialized.min_val
    assert bob.max_val == deserialized.max_val
    # assert bob.poly == deserialized.poly
    assert bob.entity_name == deserialized.entity_name
