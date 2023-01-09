# syft absolute
import syft as sy
from syft.core.adp.data_subject import DataSubject
from syft.core.adp.data_subject import dstonumpyutf8
from syft.core.adp.data_subject import numpyutf8tods
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.adp.data_subject_list import dslarraytonumpyutf8
from syft.core.adp.data_subject_list import numpyutf8todslarray

def test_data_subject_serde() -> None:
    data_subject = DataSubject("🥒pickles")

    ser = sy.serialize(dstonumpyutf8(data_subject), to_bytes=True)
    de = numpyutf8tods(sy.deserialize(ser, from_bytes=True))

    assert (de == data_subject)

def test_data_subject_array_serde() -> None:
    data_subjects = ["🥒pickles", "madhava", "short", "muchlongername", "a", "🌶"]

    data_subject_array = DataSubjectArray.from_objs(data_subjects)
    ser = sy.serialize(dslarraytonumpyutf8(data_subject_array), to_bytes=True)
    de = numpyutf8todslarray(sy.deserialize(ser, from_bytes=True))

    assert (de == data_subject_array).all()
    