# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList


def test_data_subject_list_serde() -> None:
    data_subjects = ["🥒pickles", "madhava", "short", "muchlongername", "a", "🌶"]
    data_subject_list = DataSubjectList.from_objs(data_subjects)
    ser = sy.serialize(data_subject_list, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    de.one_hot_lookup == data_subject_list.one_hot_lookup
    assert data_subject_list == de
