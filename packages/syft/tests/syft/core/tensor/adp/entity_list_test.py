# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList


def test_entity_list_serde() -> None:
    # entities = ["🥒pickles", "madhava", "short", "muchlongername", "a", "🌶"]
    # TODO: re-enable once we have some kind of long string entity mapping service
    entities = [0, 1, 2, 3, 4, 5]
    entity_list = DataSubjectList.from_objs(entities)
    ser = sy.serialize(entity_list, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    de.one_hot_lookup == entity_list.one_hot_lookup
    assert entity_list == de
