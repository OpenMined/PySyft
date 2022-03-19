# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity


def test_serde() -> None:
    ent = Entity(name="test")
    serialized = serialize(ent, to_bytes=True)
    deserialized = deserialize(serialized, from_bytes=True)

    assert ent.name == deserialized.name
