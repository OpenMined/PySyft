# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.data_subject import DataSubject


def test_serde() -> None:
    ent = DataSubject(name="test")
    serialized = serialize(ent, to_bytes=True)
    deserialized = deserialize(serialized, from_bytes=True)

    assert ent.name == deserialized.name
