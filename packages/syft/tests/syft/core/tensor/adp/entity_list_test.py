# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList
from syft.core.adp.entity import Entity


def test_entity_list_serde() -> None:
    entities = ["🥒pickles", "madhava", "short", "muchlongername", "a", "🌶"]
    entity_list = DataSubjectList.from_objs(
        [Entity(name=entity) for entity in entities]
    )
    ser = sy.serialize(entity_list, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    de.one_hot_lookup == entity_list.one_hot_lookup
    assert entity_list == de
