# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.adp.data_subject import DataSubject
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE


def test_incompatible_input_tensor_type() -> None:

    try:
        x = sy.Tensor(np.float32([1, 2, 3, 4.0]))
        x.private(min_val=0, max_val=5, data_subjects="bob")
        raise AssertionError()
    except TypeError:
        assert True


def test_string_data_subject() -> None:
    x = sy.Tensor(np.array([1], dtype=DEFAULT_INT_NUMPY_TYPE))
    out = x.private(min_val=0, max_val=5, data_subjects=["bob"])
    assert "bob" in out.child.data_subjects.one_hot_lookup


def test_list_of_strings_data_subject() -> None:
    x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
    out = x.private(min_val=0, max_val=5, data_subjects=["bob", "bob", "bob", "alice"])
    assert "bob" in out.child.data_subjects.one_hot_lookup
    assert "alice" in out.child.data_subjects.one_hot_lookup


def test_list_of_data_subject_objs() -> None:
    x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
    bob = DataSubject("bob")
    alice = DataSubject("alice")
    out = x.private(
        min_val=0,
        max_val=5,
        data_subjects=[
            bob,
            bob,
            bob,
            alice,
        ],
    )
    assert bob in out.child.data_subjects.one_hot_lookup
    assert alice in out.child.data_subjects.one_hot_lookup
