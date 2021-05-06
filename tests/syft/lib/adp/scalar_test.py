# syft absolute
from syft import deserialize
from syft import serialize
from syft.lib.adp.entity import Entity
from syft.lib.adp.scalar import Scalar


def test_scalar() -> None:
    bob = Scalar(value=1, min_val=-2, max_val=2, entity=Entity(name="Bob"))
    alice = Scalar(value=1, min_val=-1, max_val=1, entity=Entity(name="Alice"))
    bob + alice


def test_required_serde() -> None:
    bob = Scalar(value=1, min_val=-2, max_val=2, entity=Entity(name="Bob"))

    serialized = serialize(bob, to_bytes=True)
    deserialized = deserialize(serialized, from_bytes=True)

    assert bob.id == deserialized.id
    assert bob.name == deserialized.name
    assert bob.value == deserialized.value
    assert bob.min_val == deserialized.min_val
    assert bob.max_val == deserialized.max_val
    # assert bob.poly == deserialized.poly
    assert bob.entity.name == deserialized.entity.name
    assert bob.entity.id == deserialized.entity.id
    assert bob.enabled == deserialized.enabled
